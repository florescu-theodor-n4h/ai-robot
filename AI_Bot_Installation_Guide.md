# AI Bot Installation Guide

A comprehensive guide for setting up an AI bot environment with Python, Java, PyTorch/ROCm GPU support, PostgreSQL, and Tomcat.

---

## System Prerequisites

**OS:** Ubuntu 22.04 (Jammy) or similar
**GPU:** AMD (requires ROCm) or CPU fallback
**Memory:** 8GB+ recommended for ML workloads

---

## 1. Core System Setup

### Update Package Manager
```bash
sudo apt update
sudo apt upgrade -y
```

### Install Essential Build Tools
```bash
sudo apt install -y build-essential git curl wget gpg software-properties-common
```

---

## 2. Python Environment

### Add Python PPA (for Python 3.10)
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

### Install Python 3.10
```bash
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

---

## 3. GPU Support (AMD ROCm)

### Add ROCm Repository
```bash
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/rocm.gpg
echo "deb [signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0 ubuntu main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
```

### Clean Previous ROCm Installations (if upgrading)
```bash
sudo apt remove --purge -y rocm* hip* amdgpu* mesa-amdgpu* || true
sudo apt autoremove -y
sudo rm -rf /opt/rocm
```

### Install amdgpu-install
```bash
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
```

### Install ROCm Runtime and HIP Libraries
```bash
sudo amdgpu-install --usecase=rocm,hiplibsdk --no-dkms
```

### Verify ROCm Installation
```bash
rocminfo
```

---

## 4. PyTorch Installation

### Install with ROCm Support
```bash
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### Alternative: CPU-only Installation
```bash
pip install torch torchvision torchaudio
```

---

## 5. Machine Learning Libraries

Install core ML dependencies within activated venv:

```bash
pip install transformers datasets accelerate sentencepiece huggingface_hub peft einops safetensors click
```

---

## 6. Java & Maven (for JVM-based components)

### Install Java 21 JDK
```bash
sudo apt install -y openjdk-21-jdk
```

### Install Maven
```bash
sudo apt install -y maven
```

### Verify Installation
```bash
java -version
mvn -version
```

---

## 7. Tomcat Application Server

### Install Tomcat 10
```bash
sudo apt install -y tomcat10 tomcat10-admin tomcat10-docs tomcat10-examples
```

### Enable and Start Tomcat
```bash
sudo systemctl enable --now tomcat10
sudo systemctl status tomcat10
```

### Configure Tomcat Users (optional, for admin access)
```bash
sudo vim /etc/tomcat10/tomcat-users.xml
```

### Verify Tomcat is Running
```bash
ss -tulpn | grep 8080
```

---

## 8. PostgreSQL Database

### Install PostgreSQL
```bash
sudo apt install -y postgresql postgresql-contrib
```

### Enable and Start PostgreSQL
```bash
sudo systemctl enable --now postgresql
```

### Configure Network Access

Edit PostgreSQL configuration:
```bash
PGCONF=$(ls /etc/postgresql/*/main/postgresql.conf | head -n 1)
HBA=$(ls /etc/postgresql/*/main/pg_hba.conf | head -n 1)
```

Allow private network connections:
```bash
sudo sed -i "s/^#listen_addresses.*/listen_addresses = '*'/" "$PGCONF"
cat <<EOF | sudo tee -a "$HBA"

# Local dev
host    all    all    127.0.0.1/32      md5
host    all    all    ::1/128           md5

# Private networks (RFC1918)
host    all    all    10.0.0.0/8        md5
host    all    all    172.16.0.0/12     md5
host    all    all    192.168.0.0/16    md5
EOF

sudo systemctl restart postgresql
```

### Connect to PostgreSQL
```bash
sudo -u postgres psql
# Or remote:
psql -h 192.168.2.1 -p 5432 -U postgres
```

---

## 9. Network & Security Tools (optional)

### Firewall Configuration
```bash
sudo apt install -y ufw
sudo ufw allow from 192.168.0.0/16 to any port 5432 proto tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432 proto tcp
sudo ufw status verbose
```

### Network Monitoring
```bash
sudo apt install -y wireshark tshark tcpdump netcat-openbsd socat
```

---

## 10. ROCm Troubleshooting

### Fix Executable Stack Error
If you encounter: `libamdhip64.so: cannot enable executable stack as shared object requires`

```bash
# Install execstack utility
wget https://snapshot.debian.org/archive/debian/20250721T022532Z/pool/main/p/prelink/execstack_0.0.20131005-1%2Bb10_amd64.deb
mkdir -p /tmp/execstack-extract
dpkg-deb -x execstack_0.0.20131005-1+b10_amd64.deb /tmp/execstack-extract
sudo cp /tmp/execstack-extract/usr/bin/execstack /usr/local/bin/
sudo chmod 0755 /usr/local/bin/execstack

# Fix library
sudo execstack -c /opt/rocm/lib/libamdhip64.so
sudo execstack -c /opt/rocm/lib/libamdhip64.so.7
sudo execstack -c /opt/rocm/lib/libamdhip64.so.7.2.70204
```

### Clean Broken DKMS Packages
```bash
echo "Stopping DKMS/apt locks..."
sudo systemctl stop dkms 2>/dev/null || true

echo "Forcing dpkg to forget broken packages..."
sudo dpkg --purge --force-all amdgpu-dkms || true
sudo rm -f /var/lib/dpkg/info/amdgpu-dkms.*
sudo rm -rf /var/lib/dkms/amdgpu*

echo "Fixing dpkg database..."
sudo dpkg --configure -a || true
sudo apt -f install -y || true
sudo apt autoremove -y
```

---

## 11. Running Your AI Bot

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Run AI Bot Script
```bash
python run.py
```

### Create Auto-Run Script (optional)
```bash
#!/bin/bash
cd ~/AI
source AI_weight_lora/venv/bin/activate
python run.py
```

Save as `runVenv.sh` and make executable:
```bash
chmod +x runVenv.sh
./runVenv.sh
```

---

## Quick Verification Checklist

- [ ] Python 3.10 installed: `python3.10 --version`
- [ ] PyTorch working: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Java installed: `java -version`
- [ ] Maven installed: `mvn -version`
- [ ] Tomcat running: `sudo systemctl status tomcat10`
- [ ] PostgreSQL running: `sudo systemctl status postgresql`
- [ ] ROCm available: `rocminfo`

---

## Directory Structure (Expected)
```
~/AI/
├── run.py
├── runVenv.sh
├── AI_weight_lora/
│   └── venv/              # Python virtual environment
└── [your AI bot files]
```

---

## Useful Commands

### Monitor System
```bash
htop                          # Real-time system monitor
watch nvidia-smi             # GPU monitoring (if NVIDIA)
rocminfo                      # AMD GPU info
```

### Database Operations
```bash
sudo -u postgres psql        # Connect as postgres user
psql -h localhost -p 5432 -U postgres  # Connect as any user
```

### Service Management
```bash
sudo systemctl restart tomcat10
sudo systemctl restart postgresql
sudo journalctl -xeu tomcat10.service  # View service logs
```

---

## Notes

- Always activate your virtual environment before running Python scripts
- ROCm version (6.0, 6.1, 5.7) may need adjustment based on your GPU
- PostgreSQL networking requires restarting the service after config changes
- Tomcat logs are in `/var/log/tomcat10/` for debugging
- Use `sudo` for system-wide installations and service management

- a se instala pachetul pentru evitarea : ImportError: You need to install `bitsandbytes` in order to use bitsandbytes optimizers: `pip install -U bitsandbytes`
