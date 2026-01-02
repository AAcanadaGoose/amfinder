# AMFinder - amfinder_segmentation.py
#
# MIT License
# Copyright (c) 2021 Edouard Evangelisti, Carl Turner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Image Segmentation.

Crop tiles (squares) and apply diverse image modifications.

Constants
-----------
INTERPOLATION - Interpolation mode for image resizing.

Functions
------------
:function tile: Extracts a tile from a large image.
:function preprocess: Convert a tile list to NumPy array and normalise pixels.
"""

import pyvips
import random
random.seed(42)
import numpy as np

import amfinder_model as AmfModel
import amfinder_config as AmfConfig
import amfinder_log as AmfLog
from PIL import Image as PILImage



# Not the best quality, but optimized for speed.
# To check availability, type in a terminal: vips -l interpolate
INTERPOLATION = pyvips.vinterpolate.Interpolate.new('nearest')



def load(image_path, access='random'):
    """
    Loads an image using the vips library.
    """

    return pyvips.Image.new_from_file(image_path, access=access)


# --- New flexible helpers ---

def _np_to_pyvips(arr: np.ndarray) -> pyvips.Image:
    """Create a pyvips.Image from a HxWxC uint8 NumPy array."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w, c = arr.shape
    return pyvips.Image.new_from_memory(arr.tobytes(), w, h, c, format='uchar')


def _pyvips_to_np(img: pyvips.Image) -> np.ndarray:
    """Convert a pyvips.Image to a HxWxC uint8 NumPy array."""
    return np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                      shape=[img.height, img.width, img.bands])


def load_image(image_input):
    """
    Load image from flexible input format.

    Args:
        image_input: str (path), np.ndarray, PIL.Image, or pyvips.Image

    Returns:
        tuple: (pyvips.Image, np.ndarray) - both representations
    """
    if isinstance(image_input, str):
        vimg = load(image_input)
        nparr = _pyvips_to_np(vimg)
        return vimg, nparr
    elif isinstance(image_input, np.ndarray):
        nparr = image_input
        if nparr.ndim == 2:
            # grayscale -> RGB
            nparr = np.stack([nparr] * 3, axis=-1)
        vimg = _np_to_pyvips(nparr)
        return vimg, nparr
    elif isinstance(image_input, PILImage.Image):
        pil = image_input.convert('RGB')
        nparr = np.array(pil)
        vimg = _np_to_pyvips(nparr)
        return vimg, nparr
    elif isinstance(image_input, pyvips.Image):
        vimg = image_input
        nparr = _pyvips_to_np(vimg)
        return vimg, nparr
    else:
        raise TypeError('Unsupported image_input type. Expected str, np.ndarray, PIL.Image, or pyvips.Image.')


def tile_from_array(image_array: np.ndarray, r: int, c: int, edge: int = 126) -> np.ndarray:
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
    h, w = image_array.shape[:2]
    y0 = r * edge
    x0 = c * edge
    if y0 + edge > h or x0 + edge > w:
        raise ValueError('Requested tile exceeds image bounds.')
    tile = image_array[y0:y0+edge, x0:x0+edge, ...]
    # If tile size differs from model input, resize using nearest via pyvips for consistency
    if AmfModel.INPUT_SIZE != edge:
        vimg = _np_to_pyvips(tile)
        ratio = AmfModel.INPUT_SIZE / edge
        vimg = vimg.resize(ratio, interpolate=INTERPOLATION)
        tile = _pyvips_to_np(vimg)
    return tile


def preprocess_flexible(tiles_input):
    """
    Preprocess tiles from flexible input.

    Args:
        tiles_input: List of arrays, numpy array, or list of pyvips tiles

    Returns:
        np.ndarray: Preprocessed tiles (N, H, W, C)
    """
    if isinstance(tiles_input, np.ndarray):
        tiles = tiles_input
        if tiles.dtype != np.float32:
            tiles = tiles.astype(np.float32)
        return tiles / 255.0
    if isinstance(tiles_input, list):
        if len(tiles_input) == 0:
            return np.empty((0, AmfModel.INPUT_SIZE, AmfModel.INPUT_SIZE, 3), dtype=np.float32)
        # pyvips images
        if isinstance(tiles_input[0], pyvips.Image):
            arrs = [_pyvips_to_np(t) for t in tiles_input]
            arr = np.array(arrs, np.float32)
            return arr / 255.0
        # numpy arrays
        arr = np.array(tiles_input, np.float32)
        return arr / 255.0
    raise TypeError('Unsupported tiles_input type. Expected np.ndarray or list of arrays/images.')


def array_to_tiles(image_array: np.ndarray, tile_size: int = 126):
    """
    Segment image array into tiles.

    Args:
        image_array (np.ndarray): Image (H, W, C)
        tile_size (int): Tile edge length

    Returns:
        tuple: (tiles_array, nrows, ncols)
    """
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    h, w = image_array.shape[:2]
    nrows = h // tile_size
    ncols = w // tile_size
    if nrows == 0 or ncols == 0:
        AmfLog.warning(f'Tile size ({tile_size} pixels) is too large')
        return np.empty((0, tile_size, tile_size, 3), dtype=np.uint8), 0, 0
    tiles = []
    for r in range(nrows):
        for c in range(ncols):
            tiles.append(tile_from_array(image_array, r, c, tile_size))
    return np.array(tiles, dtype=np.uint8), nrows, ncols


# --- End flexible helpers ---


def tile(image, r, c, edge=None):
    """
    Extracts a tile from a large image, resizes it to
    the required CNN input image size, and applies
    data augmentation (if actve).

    :param image: The source image used to extract tiles.
    :param r: The row index of the tile to extract.
    :param c: The column index of the tile to extract.
    :return: Set of tile, converted to numpy arrays.
    :rtype: list
    """

    edge = edge if edge is not None else AmfConfig.get('tile_edge')
    tile = image.crop(c * edge, r * edge, edge, edge)

    # In super-resolution mode, ensure the tile is 42x42 pixels.
    if AmfConfig.get('super_resolution'):
      
        if edge != 42:
        
            ratio = 42 / edge
            tile = tile.resize(ratio, interpolate=INTERPOLATION)
    
    # Otherwise, use interpolation to bring tile to 126x126 pixels.
    elif AmfModel.INPUT_SIZE != edge:

        ratio = AmfModel.INPUT_SIZE / edge
        tile = tile.resize(ratio, interpolate=INTERPOLATION)

    return np.ndarray(buffer=tile.write_to_memory(),
                      dtype=np.uint8,
                      shape=[tile.height, tile.width, tile.bands])



def preprocess(tile_list):
    """
    Preprocess a list of tiles.
    
    :param tile_list: list of tiles extracted using the function above.
    :return: a numpy array containing normalised pixel values for several tiles.
    :rtype: numpy.ndarray
    """
    
    return np.array(tile_list, np.float32) / 255.0



def mosaic(image, edge=None):

    edge = edge if edge is not None else AmfConfig.get('tile_edge')

    nrows = int(image.height // edge)
    ncols = int(image.width // edge)

    if nrows == 0 or ncols == 0:

        AmfLog.warning('Tile size ({edge} pixels) is too large')
        return None
        
    else:

        tiles = []

        for r in range(nrows):

            for c in range(ncols):
 
                tiles.append(tile(image, r, c, edge))

        return tiles

