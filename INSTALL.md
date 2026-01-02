# ![](amfbrowser/data/amfbrowser.png) AMFinder installation instructions

> For a more detailed developer guide (macOS & Linux), see `dev-setup.md`.

## Python API installation (recommended)

Use AMFinder programmatically from Python.

1) Create and activate a virtual environment
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip setuptools wheel
```

2) Install dependencies
```bash
python -m pip install -r amf/requirements.txt
```
Notes
- On macOS, this installs `tensorflow-macos` and uses Metal via `tensorflow-metal`.
- System packages required: VIPS (pyvips) and Graphviz.
  - macOS: `brew install vips graphviz`
  - Ubuntu/Debian: `sudo apt install -y libvips libvips-tools graphviz`

3) Use the API in your scripts
- Ensure `amf/` is on your import path (add at runtime or set `PYTHONPATH`).
```python
import os, sys
# Put this at the very top of your script, BEFORE `from amfinder_api import AMFinder`.
sys.path.insert(0, os.path.join(os.getcwd(), 'amf'))  # make amf/ importable
from amfinder_api import AMFinder

amf = AMFinder(tile_size=126, batch_size=8)
amf.load_model('CNN1v2.h5', level=1)
df = amf.predict('Sample_photos/clear1.JPG', level=1)
print(df.head())
```
Why this line, and where does the path come from?
- It prepends the path to the `amf` folder so Python can find `amfinder_api` without packaging/installing.
- If your script is NOT in the repo root, replace `os.getcwd()` with the absolute path to the repository, e.g.:
```python
sys.path.insert(0, '/absolute/path/to/amfinder-master/amf')
```
- Or compute it relative to your script file:
```python
from pathlib import Path
REPO = Path(__file__).resolve().parents[0]  # adjust to your layout
sys.path.insert(0, str(REPO / 'amf'))
```
Other ways to link `amf` once and for all
- One-off per run:
```bash
PYTHONPATH="/absolute/path/to/amfinder-master/amf:$PYTHONPATH" python your_script.py
```
- Persist in your virtualenv: add a `.pth` file pointing to `amf`:
```bash
VENV_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "/absolute/path/to/amfinder-master/amf" > "$VENV_SITE/amfinder.pth"
```
Then you can simply:
```python
from amfinder_api import AMFinder
```

4) Optional: train models via API
```python
# X: (N,126,126,3); y: one-hot (N,3) for CNN1 / (N,4) for CNN2
model, history = amf.train(X=X, y=y, level=1, epochs=50)
```

Troubleshooting
- TensorFlow + absl import error: `ImportError: cannot import name 'command_name'`
  - Fix: `python -m pip install 'absl-py==1.4.0'`
- pyvips not found: install system VIPS (see notes above) and restart shell.

---

## Command-line tool `amf`

AMFinder also ships a CLI for batch prediction, conversion, and training.

Option A: quick setup with virtualenv (cross‑platform)
```bash
# from repo root
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r amf/requirements.txt

# run CLI
python amf/amf predict --network CNN1v2.h5 Sample_photos/clear1.JPG
```

Option B: legacy installer script
1. Move to the `amf` folder.
2. (Optional) set a specific interpreter: `export PYTHON=$(which python3)`
3. Make script executable: `chmod +x install.sh`
4. Run installer: `./install.sh`
5. Activate venv created by the script: `source amfenv/bin/activate`

Common commands
```bash
# Predict colonisation (CNN1)
python amf/amf predict --network CNN1v2.h5 Sample_photos/clear1.JPG

# Convert predictions to annotations (CNN1 default)
python amf/amf convert Sample_photos/clear1.JPG

# Predict intraradical structures (CNN2)
python amf/amf predict --network CNN2v2.h5 Sample_photos/clear1.JPG

# Convert CNN2 predictions using custom threshold
python amf/amf convert --CNN2 --threshold 0.6 Sample_photos/clear1.JPG

# Train CNN1 on images with annotations
python amf/amf train -1 -e 50 path/to/img1.jpg path/to/img2.jpg
```

Notes
- Pretrained models live in `amf/trained_networks/` and are referenced by filename (e.g., `CNN1v2.h5`).
- If you prefer a shell command `amf` in PATH, symlink `amf/amf` into a bin folder and ensure the venv is active.

---

## CPU/GPU and platform notes
- macOS: for best performance install Xcode CLT, Homebrew, then `brew install vips graphviz`.
- Linux + NVIDIA: start with CPU TF from requirements; upgrade to a CUDA‑enabled TF only after matching CUDA/cuDNN per TensorFlow’s docs.
- Older CPUs lacking AVX/AVX2 may not run official TensorFlow builds; compile from source if necessary (see TensorFlow docs).
