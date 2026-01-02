# AMFinder Developer Setup (macOS & Linux)

This guide sets up a local development environment for AMFinder using Python 3.11 and a virtual environment.

## Prerequisites
- macOS (Intel or Apple Silicon) or Linux (Ubuntu/Debian/Fedora)
- Xcode Command Line Tools (macOS)
- Homebrew (macOS) or your distro's package manager
- Python 3.11 (pyenv recommended)
- Git

---

## macOS setup

### 1) Clone the repository
```zsh
git clone https://github.com/<your-org>/amfinder.git amfinder-master
cd amfinder-master
```

### 2) Install system dependencies
AMFinder relies on VIPS (for pyvips) and Graphviz (for model plots).
```zsh
brew update
brew install vips graphviz
```

If you already have these, ensure `vips --version` prints a version.

### 3) Create and activate a virtual environment
```zsh
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip setuptools wheel
```

To deactivate later:
```zsh
deactivate
```

### 4) Install Python dependencies
Use the project requirements file in `amf/requirements.txt`.
```zsh
python -m pip install -r amf/requirements.txt
```
Notes:
- On macOS, this installs `tensorflow-macos` (and optionally uses Metal via `tensorflow-metal`).
- If you plan to train on Apple Silicon GPU, also install:
```zsh
python -m pip install tensorflow-metal
```

### 5) Known TensorFlow/absl compatibility note
Some TensorFlow builds fail to import with newer `absl-py` releases, raising:
```
ImportError: cannot import name 'command_name' from 'absl'
```
If you see this, pin a compatible version:
```zsh
python -m pip install 'absl-py==1.4.0'
```

### 6) Verify your setup
Run the provided smoke test:
```zsh
python smoke_test.py
```
You should see TensorFlow and model creation messages.

Run a CLI prediction on a sample image:
```zsh
python amf/amf predict --network CNN1v2.h5 Sample_photos/clear1.JPG
```
Expected: a ZIP with predictions saved next to the image (e.g., `Sample_photos/clear1.zip`).

Run the API test scripts:
```zsh
python tests/test_api_basic.py
python tests/test_api_cnn2.py
```

### 7) Using the Python API (example)
```zsh
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'amf'))
from amfinder_api import AMFinder

amf = AMFinder(tile_size=126, batch_size=8)
amf.load_model('CNN1v2.h5', level=1)
df = amf.predict('Sample_photos/clear1.JPG', level=1)
print(df.head())
PY
```

### 8) VS Code tips
- Select interpreter: Command Palette → Python: Select Interpreter → choose `.venv311`.
- Enable linting/formatting as needed.
- If imports show unresolved in the editor, ensure your workspace includes the `amf/` folder on `PYTHONPATH` or add `sys.path.insert(0, '<repo>/amf')` in scripts.

### Troubleshooting
- pyvips import error: ensure `brew install vips` is installed; restart your shell. On Apple Silicon, Homebrew is typically in `/opt/homebrew`.
- TensorFlow metal acceleration: install `tensorflow-metal`; fall back to CPU if unavailable.
- Graphviz errors when saving model plots: ensure `brew install graphviz` and re-open terminal.
- HDF5/model load issues: ensure `h5py` is installed (it is in requirements) and models are accessible at `amf/trained_networks/*`.

### Clean & update
```zsh
python -m pip list
python -m pip install -U pip setuptools wheel
python -m pip install -U -r amf/requirements.txt
```

### Uninstall / reset venv
```zsh
deactivate  # if active
rm -rf .venv311
```

---

## Linux setup

Commands below target Ubuntu/Debian. Adjust for your distro as noted.

### 1) Clone the repository
```bash
git clone https://github.com/<your-org>/amfinder.git amfinder-master
cd amfinder-master
```

### 2) Install system dependencies
- libvips and Graphviz are required for pyvips and plotting.
- Also install Python dev headers and build tools.

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y libvips libvips-tools graphviz python3.11 python3.11-venv python3.11-dev build-essential
```
Fedora/RHEL:
```bash
sudo dnf install -y vips vips-tools graphviz python3.11 python3.11-devel @development-tools
```
Arch:
```bash
sudo pacman -S --needed vips graphviz python python-virtualenv base-devel
```

### 3) Create and activate a virtual environment
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip setuptools wheel
```

### 4) Install Python dependencies
The requirements file is cross-platform and selects TensorFlow appropriately.
```bash
python -m pip install -r amf/requirements.txt
```
Notes:
- On Linux, this installs `tensorflow>=2.16,<3` (CPU build). For NVIDIA GPUs, use a TF version that matches your CUDA/cuDNN stack; consult TensorFlow’s install guide and pin versions as needed.

### 5) Verify your setup
```bash
python smoke_test.py
python amf/amf predict --network CNN1v2.h5 Sample_photos/clear1.JPG
```

### Troubleshooting (Linux)
- pyvips import error: ensure `libvips` is installed and visible. Check `vips --version`.
- TensorFlow import errors: many are CUDA/driver related on GPU setups; try CPU TF first. Ensure `absl-py` compatibility (see macOS note); on Linux the same fix applies:
```bash
python -m pip install 'absl-py==1.4.0'
```
- Graphviz errors: ensure `graphviz` package is installed and `dot -V` works.

---

## Quick run checklist (all platforms)
- Create venv and install `amf/requirements.txt`
- System packages installed: vips (+ tools) and graphviz
- Run:
```bash
python smoke_test.py
python amf/amf predict --network CNN1v2.h5 Sample_photos/clear1.JPG
python tests/test_api_basic.py
```

If issues persist, open an issue with OS, Python, package versions, and full error logs.
