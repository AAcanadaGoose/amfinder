# AMFinder - amfinder_api.py
# High-level Python API wrapping the existing CLI-oriented modules.

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import amfinder_config as AmfConfig
import amfinder_model as AmfModel
import amfinder_predict as AmfPredict
import amfinder_segmentation as AmfSegm
import amfinder_train as AmfTrain


class AMFinder:
    """
    Main API class for AMFinder operations.

    Attributes:
        tile_size (int): Size of tiles in pixels (default: 126)
        batch_size (int): Batch size for predictions (default: 32)
        model_cnn1: Loaded CNN1 model for colonization
        model_cnn2: Loaded CNN2 model for structures
        super_resolution (bool): Whether to use super-resolution
        config (dict): Additional configuration parameters
    """

    def __init__(self, tile_size: int = 126, batch_size: int = 32, **kwargs):
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.super_resolution = kwargs.get('super_resolution', False)
        self.config: Dict[str, Any] = dict(kwargs)
        # Lazy-loaded models
        self.model_cnn1 = None
        self.model_cnn2 = None
        # Configure module settings minimally
        AmfConfig.set('tile_edge', tile_size)
        AmfConfig.set('batch_size', batch_size)
        AmfConfig.set('super_resolution', bool(self.super_resolution))

    def load_model(self, model_path: Optional[str] = None, model: Any = None, level: int = 1):
        """
        Load a trained model.
        Args:
            model_path (str, optional): Path to .h5/.keras model file
            model (keras.Model, optional): Pre-loaded model object
            level (int): 1 for CNN1 (colonization), 2 for CNN2 (structures)
        Returns:
            keras.Model: The loaded model
        """
        if model is None and model_path is None:
            raise ValueError('model_path or model must be provided')
        mdl = AmfModel.load_flexible(model_input=model_path, model_object=model)
        if level == 1:
            self.model_cnn1 = mdl
            AmfConfig.set('level', 1)
        else:
            self.model_cnn2 = mdl
            AmfConfig.set('level', 2)
        return mdl

    def _require_model(self, level: int):
        if level == 1 and self.model_cnn1 is None:
            raise RuntimeError('CNN1 model not loaded. Call load_model(..., level=1).')
        if level == 2 and self.model_cnn2 is None:
            raise RuntimeError('CNN2 model not loaded. Call load_model(..., level=2).')
        return self.model_cnn1 if level == 1 else self.model_cnn2

    def predict(self, image: Union[str, np.ndarray, Image.Image], level: int = 1, return_format: str = 'dataframe'):
        """
        Predict fungal colonization or structures.
        image can be a file path, numpy array, PIL Image, or pyvips.Image.
        return_format: 'dataframe' | 'array' | 'dict'
        """
        model = self._require_model(level)
        vimg, npimg = AmfSegm.load_image(image)
        df = AmfPredict.predict_from_array(model, npimg, tile_size=self.tile_size,
                                           batch_size=self.batch_size, level=level)
        if return_format == 'dataframe':
            return df
        elif return_format == 'array':
            cols = AmfConfig.get('header')
            return df[cols].to_numpy()
        elif return_format == 'dict':
            return df.to_dict(orient='list')
        else:
            raise ValueError("return_format must be 'dataframe', 'array', or 'dict'")

    def predict_tiles(self, tiles: np.ndarray, level: int = 1) -> np.ndarray:
        """Predict on pre-extracted tiles array (N, H, W, C)."""
        model = self._require_model(level)
        if level == 1:
            return AmfPredict.predict_from_tiles_cnn1(model, tiles, batch_size=self.batch_size)
        return AmfPredict.predict_from_tiles_cnn2(model, tiles, batch_size=self.batch_size)

    def train(self, images=None, annotations=None, X=None, y=None,
              level: int = 1, epochs: int = 100, validation_split: float = 0.15, **kwargs):
        """
        Train a model from arrays or from images+annotations.
        Returns (trained_model, history_dict)
        """
        AmfConfig.set('level', level)
        AmfConfig.set('epochs', epochs)
        AmfConfig.set('vfrac', int(round(validation_split * 100)))
        AmfConfig.set('batch_size', kwargs.get('batch_size', self.batch_size))
        AmfConfig.set('learning_rate', kwargs.get('learning_rate', AmfConfig.get('learning_rate')))
        AmfConfig.set('patience', kwargs.get('patience', AmfConfig.get('patience')))

        if X is None or y is None:
            if images is None or annotations is None:
                raise ValueError('Provide either (X,y) arrays or (images, annotations).')
            X, y = AmfTrain.load_tiles_from_annotations(images, annotations, level=level, tile_size=self.tile_size)
        model, history = AmfTrain.train_from_arrays(
            X_train=X, y_train=y, level=level, epochs=epochs,
            batch_size=AmfConfig.get('batch_size'),
            learning_rate=AmfConfig.get('learning_rate'),
            patience=AmfConfig.get('patience'),
            data_augmentation=kwargs.get('data_augmentation', False)
        )
        if level == 1:
            self.model_cnn1 = model
        else:
            self.model_cnn2 = model
        return model, history

    def convert(self, predictions: Union[pd.DataFrame, np.ndarray], level: int = 1, threshold: float = 0.5) -> pd.DataFrame:
        """Convert predictions array to annotations DataFrame (one-hot)."""
        headers = [['Y','N','X'], ['A','V','H','I']][level-1]
        if isinstance(predictions, pd.DataFrame):
            df = predictions.copy()
            cols = [c for c in headers if c in df.columns]
            if level == 1:
                arr = df[cols].to_numpy()
                conv = np.zeros_like(arr)
                conv[np.arange(len(arr)), arr.argmax(1)] = 1
                df[cols] = conv.astype(np.uint8)
            else:
                df[cols] = (df[cols].to_numpy() >= threshold).astype(np.uint8)
            return df[['row','col'] + headers] if all(c in df.columns for c in ['row','col']) else df
        else:
            arr = np.asarray(predictions)
            if arr.ndim != 2 or arr.shape[1] != len(headers):
                raise ValueError('predictions must have shape (N, n_classes)')
            if level == 1:
                conv = np.zeros_like(arr)
                conv[np.arange(len(arr)), arr.argmax(1)] = 1
            else:
                conv = (arr >= threshold).astype(np.uint8)
            return pd.DataFrame(data=conv.astype(np.uint8), columns=headers)

    def extract_tiles(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, int, int]:
        """Extract tiles from an image input using current tile_size."""
        _, npimg = AmfSegm.load_image(image)
        return AmfSegm.array_to_tiles(npimg, tile_size=self.tile_size)

    def save_model(self, model, path: str):
        """Save a model to disk."""
        if model is None:
            raise ValueError('model must not be None')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        model.save(path)
