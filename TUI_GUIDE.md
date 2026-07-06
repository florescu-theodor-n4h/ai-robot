# LLM Agentic Dev Server - TUI Documentation

## 🎨 Beautiful Terminal User Interface

The server now includes a **colorful, newbie-friendly TUI** (Terminal User Interface) that displays:

- 🎭 **Welcome Banner** - ASCII art introduction
- ⏳ **Startup Progress** - Real-time progress indicators for each initialization stage
- 🚀 **Server Ready Screen** - Beautiful display of all server details
- ⚙️ **Environment Configuration** - Hardware and software settings table
- 📚 **API Quick Start** - Copy-paste ready API examples
- 📊 **Runtime Status** - Request tracking and error monitoring
- 🔴 **Error Display** - Beautiful error panels for troubleshooting
- 📈 **Live Status Monitor** - Real-time server metrics in a separate terminal

## 🚀 Quick Start

### Run the Server with TUI

```bash
# Activate venv
source /root/AI/enter_venv

# Run the server (TUI will display automatically)
python /root/AI/runLLMAgentForAgenticDevs.py
```

The startup will show:

```
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║       🤖  LLM AGENTIC DEV SERVER  🚀                      ║
  ║                                                           ║
  ║       Powered by TinyLlama + FastAPI + ROCm             ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝

⏳ Environment Setup....................... Configuring hardware...
✅ Environment Setup....................... Done!
⏳ Model Download.......................... Fetching TinyLlama...
✅ Model Download.......................... Downloaded 4.2GB
...
```

### Demo TUI Features

```bash
# Show welcome banner
python /root/AI/demo_tui.py welcome

# Show startup progress simulation
python /root/AI/demo_tui.py startup

# Show server ready screen
python /root/AI/demo_tui.py ready

# Show environment configuration
python /root/AI/demo_tui.py env

# Show API quick start examples
python /root/AI/demo_tui.py api

# Show error display example
python /root/AI/demo_tui.py error

# Show request tracking status
python /root/AI/demo_tui.py status

# Run all demos
python /root/AI/demo_tui.py all
```

### Run Live Status Monitor

In a **separate terminal** while the server is running:

```bash
# Activate venv
source /root/AI/enter_venv

# Run the live status monitor
python -m training_src.status_monitor

# Or from the training_src directory:
cd /root/AI && python training_src/status_monitor.py
```

This will display a live-updating dashboard showing:
- ⏱️ Server uptime
- 🤖 Model status
- 🔌 Server status
- 📬 Request count (live updates)
- ❌ Error count (live updates)
- 🔤 Total tokens generated (live updates)

## 📋 TUI Components

### 1. Welcome Banner
ASCII art greeting with server name and tech stack.

```
🤖  LLM AGENTIC DEV SERVER  🚀
Powered by TinyLlama + FastAPI + ROCm
```

### 2. Startup Progress Indicators

Shows progress for:
- ⏳ Environment Setup
- ⏳ Model Download
- ⏳ Model Initialization
- ⏳ API Setup
- ⏳ Route Registration

Each stage shows:
- `⏳` In progress
- `✅` Complete

### 3. Server Ready Screen

Displays:
- 🤖 Model repository
- 📡 Server address and port
- 🧠 Context window size
- ⚙️ Number of threads
- 🌡️ Temperature setting
- 🔤 Max tokens per response
- 📍 Available API endpoints

### 4. Environment Configuration

Shows hardware settings:
- OMP Threads
- Tokenizer Parallelism
- ROCm Status
- CUDA Status
- ROCm GFX Version

### 5. API Quick Start

Copy-paste ready examples for:
1. **Simple Chat** - `/chat` endpoint
2. **OpenAI Compatible** - `/v1/chat/completions` endpoint
3. **Health Check** - `/health` endpoint
4. **List Models** - `/v1/models` endpoint

### 6. Runtime Status Table

Tracks (displayed in both startup and live monitor):
- ⏱️ Server uptime
- 🤖 Model status
- 🔌 Server status
- 📬 Request count
- ❌ Error count
- 🔤 Total tokens generated

### 7. Live Status Monitor

A separate window that displays:
- 📊 Real-time status table
- 🕐 Timestamp of last update
- Auto-refreshing display (1 Hz)
- Can run in separate terminal

### 8. Error Display

Beautiful red panels with:
- ❌ Error title
- Error message
- Status indication

## 🎯 For Beginners

Everything is color-coded and uses emojis:

- 🟢 **Green** = Success/Complete (✅)
- 🟡 **Yellow** = In Progress (⏳)
- 🔴 **Red** = Error/Problem (❌)
- 🔵 **Blue** = Information
- 🟣 **Purple** = Settings/Configuration
- 🟠 **Orange** = Warnings/Alerts

## 💻 API Usage Examples

### Example 1: Simple Chat
```bash
curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Python?",
    "temperature": 0.6
  }'
```

### Example 2: OpenAI Compatible
```bash
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.6,
    "max_tokens": 128
  }'
```

### Example 3: Health Check
```bash
curl http://localhost:8888/health
```

### Example 4: List Models
```bash
curl http://localhost:8888/v1/models
```

## 🛠️ Configuration

The TUI displays configuration from `runLLMAgentForAgenticDevs.py`:

```python
CONFIG = Config(
    env=EnvironmentConfig(),      # Hardware settings
    model=ModelConfig(),            # Model settings
    server=ServerConfig(),           # Server settings
)
```

To customize, edit the dataclasses in `runLLMAgentForAgenticDevs.py`:

```python
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8888                # Change port here
    max_tokens: int = 512            # Change max tokens here
    temperature: float = 0.6         # Change default temperature here
    # ... etc
```

## 📊 Architecture

```
runLLMAgentForAgenticDevs.py (Source of Truth)
    ├── Config Classes (EnvironmentConfig, ModelConfig, ServerConfig)
    ├── TUI Integration (show_welcome_banner, show_startup_progress, etc.)
    └── FastAPI App Setup with TUI callbacks

training_src/
    ├── tui.py (ServerStatusTUI class)
    ├── status_monitor.py (LiveStatusMonitor for real-time tracking)
    ├── config.py (Re-exports CONFIG)
    ├── logging_setup.py (Logging)
    ├── model.py (ModelLoader)
    └── api.py (Routes)

demo_tui.py (Demo script for testing TUI features)
```

## 🎓 Key Features

✅ **Newbie Friendly**
- Colorful output with emojis
- Clear progress indicators
- Copy-paste API examples
- Beautiful error messages
- Live status monitoring

✅ **Informative**
- Shows all configuration at startup
- Displays server details
- Tracks requests and errors in real-time
- Live-updating status dashboard

✅ **Professional**
- Clean, organized layout
- Proper borders and spacing
- Consistent color scheme
- ASCII art styling

## 🚀 Startup Sequence

When you run the server, you'll see:

1. 🎭 Welcome banner
2. ⏳ Environment configuration progress
3. ⏳ Model download progress
4. ⏳ Model initialization progress
5. ⏳ API setup progress
6. ⏳ Route registration progress
7. 🚀 **Server Ready** screen
8. ⚙️ Environment details
9. 📚 API quick start examples

Then the server runs and logs all requests.

## 📝 Files

- **`runLLMAgentForAgenticDevs.py`** - Main server with TUI integration
- **`training_src/tui.py`** - ServerStatusTUI class
- **`training_src/status_monitor.py`** - LiveStatusMonitor for real-time tracking
- **`training_src/tui_compat.py`** - Re-export compatibility layer
- **`demo_tui.py`** - Demo script for testing TUI features

## 🎨 Color Reference

| Color | Meaning | Use |
|-------|---------|-----|
| 🟢 Green | Success | Completed tasks, running server |
| 🟡 Yellow | In Progress | Loading, waiting, processing |
| 🔴 Red | Error | Failures, errors, warnings |
| 🔵 Cyan | Info | Titles, headers, important text |
| 🟣 Magenta | Config | Settings, configuration values |
| 🟠 Orange | Action | Examples, call-to-action |

## ❓ Troubleshooting

**Issue: TUI not displaying colors**
- Your terminal may not support ANSI colors
- Try a different terminal application
- Rich should auto-detect and fallback gracefully

**Issue: TUI is misaligned**
- Make your terminal window wider
- Rich adjusts to terminal width automatically

**Issue: Server starts but TUI doesn't show**
- Check that `rich` is installed: `pip install rich`
- Check logs for any import errors

**Issue: Live status monitor crashes**
- Make sure you're using separate terminals for server and monitor
- Check that `rich` is properly installed
- Try running the demo first: `python demo_tui.py all`

## 📚 Learn More

- **FastAPI** - https://fastapi.tiangolo.com/
- **Rich** - https://rich.readthedocs.io/
- **Llama.cpp** - https://github.com/ggerganov/llama.cpp
- **TinyLlama** - https://github.com/jzhang38/TinyLlama
