# AMFinder Python API Implementation Specification

## Overview

Transform the AMFinder command-line tool into a modular Python library that can be imported and used programmatically. The library should support both file-based operations (maintaining backward compatibility) and in-memory operations using numpy arrays and pandas DataFrames.

## Goals

1. Enable AMFinder to be imported as a Python module in other projects
2. Support direct input/output of data using numpy and pandas structures
3. Maintain maximum flexibility for different usage contexts
4. Preserve backward compatibility with existing CLI functionality
5. Separate business logic from I/O operations

## Architecture Decisions

### Configuration Strategy
- Use **instance-based configuration** with the `AMFinder` class
- Store configuration as instance attributes
- Allow method-level parameter overrides for flexibility
- Provide sensible defaults matching current CLI behavior

### Data Format Standardization

**Images:**
- Numpy arrays with shape `(height, width, channels)` for single images
- Numpy arrays with shape `(n_tiles, height, width, channels)` for tile batches
- Values should be uint8 in range [0, 255]
- Support 3-channel RGB images

**Predictions:**
- Return pandas DataFrames matching current TSV structure
- For CNN1: columns = ['row', 'col', 'Y', 'N', 'X'] (colonized, non-colonized, background)
- For CNN2: columns = ['row', 'col', 'A', 'V', 'H', 'I'] (arbuscules, vesicles, hyphopodia, intraradical hyphae)

**Training Data:**
- Accept numpy arrays: `X` (tiles) and `y` (labels)
- For CNN1: `y` shape = `(n_samples, 3)` one-hot encoded
- For CNN2: `y` shape = `(n_samples, 4)` one-hot encoded
- Return training history as dictionary and model object

### Input/Output Flexibility

Support **multiple input formats** with automatic detection:
- File paths (strings): `"/path/to/image.jpg"`
- Numpy arrays: `np.ndarray`
- PIL Image objects: `PIL.Image.Image`
- PyVips images: `pyvips.Image`
- Pandas DataFrames (for annotations)
- Dictionary format for batch operations

Detection logic should use type checking:
```python
if isinstance(input, str):
    # Load from file
elif isinstance(input, np.ndarray):
    # Use directly
elif isinstance(input, Image.Image):
    # Convert from PIL
elif isinstance(input, pyvips.Image):
    # Use pyvips operations
```

## Implementation Plan

### 1. Create Core API Module: `amfinder_api.py`

Create a new file `amf/amfinder_api.py` containing the main `AMFinder` class.

**Class Structure:**

```python
class AMFinder:
    """
    Main API class for AMFinder operations.
    
    Attributes:
        tile_size (int): Size of tiles in pixels (default: 126)
        batch_size (int): Batch size for predictions (default: 32)
        model_cnn1 (keras.Model): Loaded CNN1 model for colonization
        model_cnn2 (keras.Model): Loaded CNN2 model for structures
        super_resolution (bool): Whether to use super-resolution
        config (dict): Additional configuration parameters
    """
    
    def __init__(self, tile_size=126, batch_size=32, **kwargs):
        """Initialize AMFinder instance with configuration."""
        pass
    
    def load_model(self, model_path=None, model=None, level=1):
        """
        Load a trained model.
        
        Args:
            model_path (str, optional): Path to .h5 model file
            model (keras.Model, optional): Pre-loaded model object
            level (int): 1 for CNN1 (colonization), 2 for CNN2 (structures)
        
        Returns:
            keras.Model: The loaded model
        """
        pass
    
    def predict(self, image, level=1, return_format='dataframe'):
        """
        Predict fungal colonization or structures.
        
        Args:
            image: Image input (flexible format - see below)
            level (int): 1 for colonization, 2 for structures
            return_format (str): 'dataframe', 'array', or 'dict'
        
        Image input formats:
            - str: File path to image
            - np.ndarray: Image array (H, W, C)
            - PIL.Image: PIL Image object
            - pyvips.Image: PyVips image object
        
        Returns:
            pd.DataFrame or np.ndarray or dict: Predictions
        """
        pass
    
    def predict_tiles(self, tiles, level=1):
        """
        Predict on pre-extracted tiles.
        
        Args:
            tiles (np.ndarray): Tiles array (N, H, W, C)
            level (int): 1 for colonization, 2 for structures
        
        Returns:
            np.ndarray: Predictions array
        """
        pass
    
    def train(self, images=None, annotations=None, X=None, y=None, 
              level=1, epochs=100, validation_split=0.15, **kwargs):
        """
        Train a model.
        
        Args:
            images (list, optional): List of image inputs (flexible format)
            annotations (list, optional): List of annotation DataFrames or paths
            X (np.ndarray, optional): Pre-extracted tiles array
            y (np.ndarray, optional): Labels array (one-hot encoded)
            level (int): 1 for CNN1, 2 for CNN2
            epochs (int): Number of training epochs
            validation_split (float): Validation fraction
            **kwargs: Additional training parameters
        
        Returns:
            tuple: (trained_model, history_dict)
        """
        pass
    
    def convert(self, predictions, level=1, threshold=0.5):
        """
        Convert predictions to annotations.
        
        Args:
            predictions (pd.DataFrame or np.ndarray): Prediction probabilities
            level (int): 1 for CNN1, 2 for CNN2
            threshold (float): Threshold for CNN2 conversion
        
        Returns:
            pd.DataFrame: Annotations (one-hot encoded)
        """
        pass
    
    def extract_tiles(self, image):
        """
        Extract tiles from an image.
        
        Args:
            image: Image input (flexible format)
        
        Returns:
            tuple: (tiles_array, nrows, ncols)
        """
        pass
    
    def save_model(self, model, path):
        """Save a model to disk."""
        pass
```

### 2. Refactor Image Segmentation Module: `amfinder_segmentation.py`

Add new functions to handle multiple input formats:

```python
def load_image(image_input):
    """
    Load image from flexible input format.
    
    Args:
        image_input: str (path), np.ndarray, PIL.Image, or pyvips.Image
    
    Returns:
        tuple: (pyvips.Image, np.ndarray) - both representations
    """
    pass

def tile_from_array(image_array, r, c, edge=126):
    """
    Extract tile from numpy array.
    
    Args:
        image_array (np.ndarray): Image array (H, W, C)
        r (int): Row index
        c (int): Column index
        edge (int): Tile size
    
    Returns:
        np.ndarray: Tile array (edge, edge, C)
    """
    pass

def preprocess_flexible(tiles_input):
    """
    Preprocess tiles from flexible input.
    
    Args:
        tiles_input: List of arrays, numpy array, or list of pyvips tiles
    
    Returns:
        np.ndarray: Preprocessed tiles (N, H, W, C)
    """
    pass

def array_to_tiles(image_array, tile_size=126):
    """
    Segment image array into tiles.
    
    Args:
        image_array (np.ndarray): Image (H, W, C)
        tile_size (int): Tile edge length
    
    Returns:
        tuple: (tiles_array, nrows, ncols)
    """
    pass
```

### 3. Refactor Prediction Module: `amfinder_predict.py`

Extract core prediction logic from file I/O:

```python
def predict_from_tiles_cnn1(model, tiles, batch_size=32):
    """
    Predict colonization from tile array.
    
    Args:
        model (keras.Model): CNN1 model
        tiles (np.ndarray): Preprocessed tiles (N, 126, 126, 3)
        batch_size (int): Batch size
    
    Returns:
        np.ndarray: Predictions (N, 3) - Y, N, X probabilities
    """
    pass

def predict_from_tiles_cnn2(model, tiles, batch_size=32):
    """
    Predict structures from tile array.
    
    Args:
        model (keras.Model): CNN2 model
        tiles (np.ndarray): Preprocessed tiles (N, 126, 126, 3)
        batch_size (int): Batch size
    
    Returns:
        np.ndarray: Predictions (N, 4) - A, V, H, I probabilities
    """
    pass

def predict_from_array(model, image_array, tile_size=126, batch_size=32, level=1):
    """
    End-to-end prediction from image array.
    
    Args:
        model (keras.Model): Trained model
        image_array (np.ndarray): Image (H, W, C)
        tile_size (int): Tile size
        batch_size (int): Batch size
        level (int): 1 or 2
    
    Returns:
        pd.DataFrame: Predictions with row/col coordinates
    """
    pass

def format_predictions(predictions, nrows, ncols, level=1):
    """
    Format prediction array as DataFrame with coordinates.
    
    Args:
        predictions (np.ndarray): Prediction probabilities
        nrows (int): Number of tile rows
        ncols (int): Number of tile columns
        level (int): 1 or 2
    
    Returns:
        pd.DataFrame: Formatted predictions
    """
    pass

# Keep existing run() function for CLI compatibility
def run(input_files):
    """Existing CLI entry point - unchanged."""
    pass
```

### 4. Refactor Training Module: `amfinder_train.py`

Separate data loading from training logic:

```python
def train_from_arrays(X_train, y_train, X_val=None, y_val=None, 
                     level=1, epochs=100, batch_size=32, 
                     learning_rate=0.001, patience=12, 
                     data_augmentation=False, **kwargs):
    """
    Train model from numpy arrays.
    
    Args:
        X_train (np.ndarray): Training tiles (N, 126, 126, 3)
        y_train (np.ndarray): Training labels (N, n_classes)
        X_val (np.ndarray, optional): Validation tiles
        y_val (np.ndarray, optional): Validation labels
        level (int): 1 for CNN1, 2 for CNN2
        epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        patience (int): Early stopping patience
        data_augmentation (bool): Use data augmentation
        **kwargs: Additional parameters
    
    Returns:
        tuple: (model, history_dict)
    """
    pass

def load_tiles_from_annotations(image_inputs, annotation_inputs, level=1):
    """
    Load and extract tiles from images with annotations.
    
    Args:
        image_inputs (list): List of image inputs (flexible format)
        annotation_inputs (list): List of DataFrames or ZIP file paths
        level (int): 1 or 2
    
    Returns:
        tuple: (X_array, y_array) - tiles and labels
    """
    pass

def load_annotations_flexible(annotation_input, level=1):
    """
    Load annotations from flexible input.
    
    Args:
        annotation_input: DataFrame, ZIP path, or TSV path
        level (int): 1 or 2
    
    Returns:
        pd.DataFrame: Annotations
    """
    pass

# Keep existing run() function for CLI compatibility
def run(input_files):
    """Existing CLI entry point - unchanged."""
    pass
```

### 5. Extend Model Module: `amfinder_model.py`

Add flexible model loading/saving:

```python
def load_flexible(model_input):
    """
    Load model from flexible input.
    
    Args:
        model_input: str (path), keras.Model object, or dict (weights)
    
    Returns:
        keras.Model: Loaded model
    """
    pass

def save_model_to_bytes(model):
    """
    Serialize model to bytes (for in-memory operations).
    
    Args:
        model (keras.Model): Model to serialize
    
    Returns:
        bytes: Serialized model
    """
    pass

def load_model_from_bytes(model_bytes):
    """
    Load model from bytes.
    
    Args:
        model_bytes (bytes): Serialized model
    
    Returns:
        keras.Model: Loaded model
    """
    pass

# Keep existing load() function, enhance with flexible input
def load(network=None, model_object=None, level=1):
    """
    Enhanced load function supporting multiple inputs.
    
    Args:
        network (str, optional): Path to model file
        model_object (keras.Model, optional): Pre-loaded model
        level (int): 1 or 2
    
    Returns:
        keras.Model: Loaded model
    """
    pass
```

### 6. Create Compatibility Module: `amfinder_compat.py`

Bridge between API and CLI:

```python
"""
Compatibility layer between AMFinder API and CLI.
Adapters to maintain backward compatibility.
"""

def api_to_cli_config(amfinder_instance):
    """
    Convert AMFinder instance config to CLI PAR dict.
    
    Args:
        amfinder_instance (AMFinder): API instance
    
    Returns:
        dict: Configuration in CLI format
    """
    pass

def cli_predict_wrapper(input_files):
    """
    Wrapper for CLI predict using API backend.
    
    Args:
        input_files (list): Input file paths
    """
    pass

def cli_train_wrapper(input_files):
    """
    Wrapper for CLI train using API backend.
    
    Args:
        input_files (list): Input file paths
    """
    pass
```

### 7. Update Main CLI Entry Point: `amf`

Optionally update to use new API internally (for cleaner code):

```python
# Keep existing structure, optionally refactor internals to use AMFinder class
# This is optional - can maintain current implementation for backward compatibility

def main():
    AmfConfig.initialize()
    run_mode = AmfConfig.get('run_mode')
    input_files = AmfConfig.get_input_files()
    
    # Existing code continues to work...
    # Or optionally: use AMFinder API internally
```

## Usage Examples

### Example 1: Prediction from File

```python
from amfinder_api import AMFinder

# Initialize
amf = AMFinder(tile_size=126, batch_size=32)

# Load model
amf.load_model('trained_networks/CNN1v2.h5', level=1)

# Predict from file
predictions = amf.predict('image.jpg', level=1)

# predictions is a DataFrame with columns: row, col, Y, N, X
print(predictions.head())
```

### Example 2: Prediction from Numpy Array

```python
import numpy as np
from PIL import Image
from amfinder_api import AMFinder

# Load image as numpy array
img = Image.open('image.jpg')
img_array = np.array(img)

# Initialize and predict
amf = AMFinder()
amf.load_model('trained_networks/CNN1v2.h5', level=1)
predictions = amf.predict(img_array, level=1)
```

### Example 3: Training with Numpy Arrays

```python
from amfinder_api import AMFinder
import numpy as np

# Prepare training data
X_train = np.load('tiles.npy')  # Shape: (N, 126, 126, 3)
y_train = np.load('labels.npy')  # Shape: (N, 3) for CNN1

# Train
amf = AMFinder()
model, history = amf.train(
    X=X_train, 
    y=y_train, 
    level=1, 
    epochs=100,
    validation_split=0.15
)

# Save model
amf.save_model(model, 'my_model.h5')
```

### Example 4: Training from Files (Backward Compatible)

```python
from amfinder_api import AMFinder

amf = AMFinder()
model, history = amf.train(
    images=['img1.jpg', 'img2.jpg'],
    annotations=['img1.zip', 'img2.zip'],  # ZIP files with annotations
    level=1,
    epochs=100
)
```

### Example 5: Extract Tiles Only

```python
from amfinder_api import AMFinder

amf = AMFinder(tile_size=126)
tiles, nrows, ncols = amf.extract_tiles('image.jpg')

# tiles is numpy array (N, 126, 126, 3)
print(f"Extracted {tiles.shape[0]} tiles ({nrows} x {ncols})")
```

### Example 6: Predict on Pre-extracted Tiles

```python
from amfinder_api import AMFinder
import numpy as np

# Assume you have tiles from another source
tiles = np.load('my_tiles.npy')  # (N, 126, 126, 3)

amf = AMFinder()
amf.load_model('trained_networks/CNN1v2.h5', level=1)
predictions = amf.predict_tiles(tiles, level=1)

# predictions is numpy array (N, 3)
```

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - Create `amfinder_api.py` with class skeleton
   - Add flexible input handling to `amfinder_segmentation.py`
   - Update `amfinder_model.py` for flexible model loading

2. **Phase 2: Prediction Path**
   - Refactor `amfinder_predict.py` to separate logic from I/O
   - Implement `AMFinder.predict()` and `AMFinder.predict_tiles()`
   - Test prediction workflows with different input formats

3. **Phase 3: Training Path**
   - Refactor `amfinder_train.py` to separate logic from I/O
   - Implement `AMFinder.train()` with flexible inputs
   - Test training workflows

4. **Phase 4: Utilities & Compatibility**
   - Implement `AMFinder.convert()` and other utility methods
   - Create `amfinder_compat.py` for CLI compatibility
   - Add comprehensive docstrings and type hints

5. **Phase 5: Testing & Documentation**
   - Write unit tests for API methods
   - Test backward compatibility with CLI
   - Create API documentation and examples

## Testing Strategy

1. **Unit Tests**: Test each new function individually
2. **Integration Tests**: Test AMFinder class methods end-to-end
3. **Compatibility Tests**: Ensure CLI still works unchanged
4. **Format Tests**: Test with all supported input formats (str, np.ndarray, PIL.Image, pyvips.Image)
5. **Edge Cases**: Test with various image sizes, tile sizes, batch sizes

## Notes for Implementation

- **Minimize Changes to Existing Code**: Keep existing functions working; add new functions alongside them
- **Type Checking**: Use `isinstance()` for format detection, not try/except
- **Error Handling**: Provide clear error messages for unsupported formats
- **Memory Efficiency**: For large images, consider memory-mapped arrays or chunked processing
- **Thread Safety**: Consider if AMFinder instances need to be thread-safe
- **Documentation**: Add comprehensive docstrings with type hints (Google or NumPy style)
- **Dependencies**: Document any new dependencies (should be minimal - mainly leverage existing numpy/pandas/PIL)
