#!/bin/bash ./runVenv.sh

import os
import sys
import subprocess
import importlib.util

# =========================
# ROCm ENV HARDENING
# =========================

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["PYTORCH_ROCM_ARCH"] = "gfx1034"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"


# =========================
# PACKAGE CHECKER
# =========================

def is_installed(pkg):
    return importlib.util.find_spec(pkg) is not None

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def ensure(pkg, pip_name=None):
    if not is_installed(pkg):
        print(f"[INSTALL] {pkg}")
        pip_install(pip_name or pkg)
    else:
        print(f"[OK] {pkg}")


ensure("torch")
ensure("transformers")
ensure("datasets")
ensure("peft")
ensure("accelerate")
ensure("sentencepiece")
ensure("huggingface_hub")


# =========================
# IMPORTS
# =========================

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from huggingface_hub import snapshot_download


print("\n[CHECK] PyTorch version:", torch.__version__)

rocm_available = torch.version.hip is not None

if rocm_available:
    print("[ROCM] detected HIP:", torch.version.hip)
else:
    print("[WARNING] ROCm not detected")


# =========================
# MODEL
# =========================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("\n[MODEL] loading...")
model_path = snapshot_download(repo_id=model_id)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# IMPORTANT FIX: pad token missing in many LLaMA models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


dtype = torch.float16 if (torch.cuda.is_available() or rocm_available) else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=dtype
)


# =========================
# LORA (ADVANCED CONFIGURATION)
# =========================

lora_config = LoraConfig(
    r=16,                                    # Increased rank for better capacity
    lora_alpha=32,                           # Increased alpha (2x r)
    target_modules=[
        "q_proj", "v_proj",                  # Query and Value projections
        "k_proj", "o_proj",                  # Key and Output projections (advanced)
        "up_proj", "down_proj"               # Feed-forward layers (advanced)
    ],
    lora_dropout=0.1,                        # Increased dropout for regularization
    bias="lora_only",                        # Apply LoRA to bias as well (advanced)
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    
    # Advanced PEFT features
    modules_to_save=None,                    # All modules use LoRA
    fan_in_fan_out=False,
    
    # LoRA+ (advanced optimization)
    use_rslora=True,                         # Use rank-stabilized LoRA for stability
    
    # Init settings for better convergence
    init_lora_weights=True,
)

model = get_peft_model(model, lora_config)


# =========================
# DATASET (FIXED: ADD LABELS)
# =========================

data = [
    {"text": "CV: Python ML engineer. Job: AI engineer. Match: high."},
    {"text": "CV: Designer. Job: backend engineer. Match: low."},
    {"text": "CV: Backend developer 5 years. Job: Python API engineer. Match: high."}
]

dataset = Dataset.from_list(data)


def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # CRITICAL FIX: causal LM needs labels
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = dataset.map(tokenize)
dataset.set_format(type="torch")


# =========================
# COLLATOR (IMPORTANT FIX)
# =========================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# =========================
# TRAINING (ADVANCED SETTINGS)
# =========================

args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,                      # Increased epochs for better convergence
    learning_rate=5e-4,                      # Optimized learning rate for advanced LoRA
    lr_scheduler_type="cosine",              # Cosine annealing for better convergence (advanced)
    warmup_steps=100,                        # Warmup period (advanced)
    warmup_ratio=0.1,                        # Warmup ratio
    fp16=(dtype == torch.float16),
    logging_steps=1,
    save_strategy="steps",                   # Save checkpoint strategy
    save_steps=50,
    eval_strategy="no",                      # Can add evaluation later
    gradient_checkpointing=True,             # Memory optimization
    optim="paged_adamw_32bit",               # Better optimizer for LoRA (advanced)
    weight_decay=0.01,                       # L2 regularization
    max_grad_norm=0.3,                       # Gradient clipping for stability
    report_to=[]
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator
)


print("[TRAIN] starting...")
trainer.train()


# =========================
# SAVE
# =========================

model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

print("\n[DONE] saved to ./lora_adapter")


# =========================
# SYSTEM SUMMARY
# =========================

print("\n=== SYSTEM SUMMARY ===")
print("ROCm:", rocm_available)
print("Device:", "GPU/ROCm or CPU")
print("Mode:", "LoRA fine-tuning")
