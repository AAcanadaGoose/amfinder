# AMFinder migration to modern TensorFlow/Keras and Python 3.11

This document summarizes the code and dependency changes applied to run AMFinder on current macOS with Python 3.11 and TensorFlow 2.16 (tf.keras), and explains how to set up and use the system now.

## What changed

Codebase
- Unified on tf.keras (removed standalone `keras` imports):
  - `amf/amfinder_train.py`
    - Replaced `import keras` with `from tensorflow import keras`.
    - Replaced `from keras.callbacks ...` with `from tensorflow.keras.callbacks ...`.
    - Replaced `from keras.preprocessing.image ...` with `from tensorflow.keras.preprocessing.image ...`.
    - Kept `keras.utils.plot_model` via `from tensorflow import keras`.
    - Added `from PIL import Image` for tile saving.
  - `amf/amfinder_model.py`
    - Switched to `tensorflow.keras` for `Model`, `Layers`, `Adam`, `he_uniform`.
    - Metrics from `acc` to `accuracy`.
    - Load legacy `.h5` with `compile=False` for compatibility:
      - `keras.models.load_model(path, compile=False)`.
  - `amf/amfinder_superresolution.py`
    - Switched to `tensorflow.keras` for all layers/models/backend/optimizer.
    - Removed `tf-agents` `Network` wrapper. Now uses a non-trainable `Model` wrapper for the discriminator:
      - `frozen_discriminator = Model(discriminator.inputs, discriminator.outputs, name='frozen_discriminator'); frozen_discriminator.trainable = False`.
  - `amf/amfinder_calc.py`
    - Already using TensorFlow 2 eager APIs and tf.keras; no change required.

Dependencies
- `amf/requirements.txt` modernized:
  - On macOS: `tensorflow-macos>=2.16,<3` and `tensorflow-metal>=1.1`.
  - Elsewhere: `tensorflow>=2.16,<3`.
  - Kept/updated: `numpy`, `scipy`, `h5py`, `pandas`, `matplotlib`, `Pillow`, `opencv-python`, `pyvips`, `PyYAML`, `psutil`, `scikit-learn`.
  - Optional: `pydot`, `graphviz`, `tensorboard`.
  - Added missing: `imagesize` (used by `amfinder_convert.py`).
  - Removed legacy/unused: standalone `keras`, `Keras-Preprocessing`, `tf-agents`, `tensorflow-probability`, `gym`, `dm-tree`, `gin-config`, `google-*`.

Utility
- Added `smoke_test.py` to validate TensorFlow and model construction.

## Why these changes
- Modern TensorFlow bundles Keras as `tf.keras`. Mixing with standalone `keras` causes version conflicts and import/runtime errors.
- TensorFlow wheels aren’t available for Python 3.13 on macOS at this time; Python 3.11 is supported and recommended.
- `tf-agents.Network` was previously used to silence warnings; the plain `tf.keras.Model` wrapper is the supported approach.
- Loading legacy `.h5` models may require `compile=False` because they were saved under older Keras; recompilation can be done after loading if needed.

## Set up the environment (macOS, Python 3.11)

1) Install Python 3.11 (Homebrew):
```bash
brew install python@3.11
```

2) Create and activate a virtual environment:
```bash
cd /path/to/amfinder-master
/opt/homebrew/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python -V
```

3) Install project requirements:
```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r amf/requirements.txt
```

4) (Recommended) System packages for extras:
```bash
brew install vips graphviz   # for pyvips and plot_model
```

5) Verify the setup (optional):
```bash
python smoke_test.py
```
Expected: prints TensorFlow version (2.16.x), confirms tf.keras, imports project modules, and builds `create_cnn1()`.

## How to run AMFinder now

Activate the environment each time:
```bash
source .venv311/bin/activate
```

Show CLI help:
```bash
python amf/amf --help
```

Predict fungal colonisation (CNN1) using a pre-trained model:
```bash
python amf/amf predict --network CNN1v2.h5 path/to/image1.jpg path/to/image2.jpg
```

Convert CNN1 predictions to annotations:
```bash
python amf/amf convert --CNN1 path/to/image1.jpg path/to/image2.jpg
```

Predict intraradical structures (CNN2), then convert with a threshold:
```bash
python amf/amf predict --network CNN2v2.h5 path/to/image1.jpg
python amf/amf convert --CNN2 --threshold 0.6 path/to/image1.jpg
```

Train a new model (examples):
```bash
python amf/amf train --CNN1 --epochs 50 --batch_size 32 path/to/*.jpg
# or
python amf/amf train --CNN2 --epochs 50 --batch_size 32 path/to/*.jpg
```

## Using the pre-trained networks
- Location: `amf/trained_networks/` contains `CNN1v1.h5`, `CNN1v2.h5`, `CNN2v1.h5`, `CNN2v2.h5`.
- Selection: Pass `--network <file>` to the CLI in predict mode, e.g. `--network CNN1v2.h5`.
- How they are loaded in code:
  - `amf/amfinder_model.py::load()` resolves the model path to `amf/trained_networks/<file>` via `amfinder_config.get_appdir()` and loads it with `tf.keras.models.load_model(..., compile=False)`.
  - After loading, the code detects whether the model is CNN1 or CNN2 by `model.name` and sets the working level accordingly.
- Compatibility: Legacy `.h5` files load under tf.keras. New models can be saved in the modern `.keras` format if desired.

## GPU/Metal acceleration (macOS)
- Installed `tensorflow-metal` enables Apple GPU acceleration. The log will show a device like `METAL` (Apple M-series).
- The initial “0 MB memory” message is normal before tensors are allocated.

## Notes and common pitfalls
- FutureWarning about `np.object` from Keras export utils is harmless.
- If `pyvips` import fails at runtime, ensure `brew install vips`.
- `plot_model` needs Graphviz + pydot: `brew install graphviz`.
- On Linux/Windows, the platform markers in `requirements.txt` will install `tensorflow` instead of `tensorflow-macos`.
- If you later need ONNX export, consider installing compatible `tf2onnx`/`onnx` versions separately (not required for AMFinder core).

## Files touched
- Updated:
  - `amf/amfinder_train.py`
  - `amf/amfinder_model.py`
  - `amf/amfinder_superresolution.py`
  - `amf/requirements.txt`
- Added:
  - `smoke_test.py`
  - This `MIGRATION.md`

## Questions
If anything doesn’t run as expected, share the full console output and your command. I’ll help diagnose quickly.
