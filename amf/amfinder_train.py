# AMFinder - amfinder_train.py
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
Neural network training.

Trains a convolutional neural network with a set of ink-stained root
images associated with tile annotations which label colonised root
sections (CNN1) or intraradical hyphal structures (CNN2).
Annotations are stored in an auxiliary ZIP archive.

Class
------------
:class ImageDataGeneratorMO:
    Custom data generator for multiple single-variable outputs.
    Reference: https://github.com/keras-team/keras/issues/3761

Functions
------------
:function get_zipfile: Returns the path of an auxiliary ZIP archive.
:function import_settings: Imports image settings from a ZIP archive.
:function import_annotations: Imports tile annotations from a ZIP archive.
:function estimate_background_subsampling: Estimates background subsampling.
:function load_dataset: Loads training dataset.
:function class_weights: Computes class weights
:function get_callbacks: Configures Keras callbacks.
:function save_model_architecture: Saves neural network architecture.
:function print_memory_usage: Prints memory used to load training data.
:function run: Runs a training session.
"""

import io
import os
#import cv2
import yaml
from tensorflow import keras
import psutil
import random
random.seed(42)
import pyvips
import operator
import functools
import amfinder_zipfile as zf
import numpy as np
import pandas as pd
from PIL import Image

from contextlib import redirect_stdout

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import amfinder_log as AmfLog
import amfinder_plot as AmfPlot
import amfinder_save as AmfSave
import amfinder_image as AmfImage
import amfinder_model as AmfModel
import amfinder_config as AmfConfig
import amfinder_segmentation as AmfSegm



def get_zipfile(path):
    """
    Returns the path of the auxiliary ZIP archive associated
    with the given image.

    :param path: Path to an input image.
    :return: Path to the corresponding auxiliary ZIP archive.
    :rtype: string
    """

    return '{}.zip'.format(os.path.splitext(path)[0])


# --- New modular training helpers for API use ---

def load_annotations_flexible(annotation_input, level=1):
    """
    Load annotations from flexible input.

    Args:
        annotation_input: DataFrame, ZIP path, or TSV path
        level (int): 1 or 2

    Returns:
        pd.DataFrame: Annotations with columns ['row','col', classes...]
    """
    if isinstance(annotation_input, pd.DataFrame):
        return annotation_input
    if isinstance(annotation_input, str):
        path = annotation_input
        # ZIP file
        if path.lower().endswith('.zip') and zf.is_zipfile(path):
            with zf.ZipFile(path, 'r') as z:
                base = 'col' if level == 1 else 'myc'
                table = f'{base}.tsv'
                if table not in z.namelist():
                    return None
                raw = z.read(table).decode('utf-8')
                return pd.read_csv(io.StringIO(raw), sep='\t')
        # TSV file
        if path.lower().endswith('.tsv') and os.path.isfile(path):
            return pd.read_csv(path, sep='\t')
    return None


def load_tiles_from_annotations(image_inputs, annotation_inputs, level=1, tile_size=126):
    """
    Load and extract tiles from images with annotations.

    Args:
        image_inputs (list): List of image inputs (flexible format)
        annotation_inputs (list): List of DataFrames or ZIP/TSV file paths
        level (int): 1 or 2

    Returns:
        tuple: (X_array, y_array) - tiles and labels
    """
    if image_inputs is None or annotation_inputs is None:
        return None, None
    if len(image_inputs) != len(annotation_inputs):
        raise ValueError('images and annotations lists must have same length')
    X_tiles = []
    y_labels = []
    headers = [['Y','N','X'], ['A','V','H','I']][level-1]
    for img_in, ann_in in zip(image_inputs, annotation_inputs):
        vimg, npimg = AmfSegm.load_image(img_in)
        ann = load_annotations_flexible(ann_in, level=level)
        if ann is None:
            continue
        for row in ann.itertuples(index=False):
            r = int(getattr(row, 'row'))
            c = int(getattr(row, 'col'))
            tile = AmfSegm.tile_from_array(npimg, r, c, edge=tile_size)
            X_tiles.append(tile)
            y_vec = [int(getattr(row, h)) for h in headers]
            y_labels.append(y_vec)
    if len(X_tiles) == 0:
        return np.empty((0, tile_size, tile_size, 3), dtype=np.uint8), np.empty((0, len(headers)), dtype=np.uint8)
    return np.array(X_tiles, dtype=np.uint8), np.array(y_labels, dtype=np.uint8)


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

    Returns:
        tuple: (model, history_dict)
    """
    # Configure global settings to reuse convenience utilities
    AmfConfig.set('level', level)
    AmfConfig.set('learning_rate', learning_rate)
    AmfConfig.set('patience', patience)
    AmfConfig.set('batch_size', batch_size)

    # Build model
    if level == 1:
        model = AmfModel.create_cnn1()
    else:
        model = AmfModel.create_cnn2()

    # Prepare generators
    if level == 1:
        if data_augmentation:
            t_gen = ImageDataGenerator(rescale=1.0/255,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       brightness_range=[0.75, 1.25])
        else:
            t_gen = ImageDataGenerator(rescale=1.0/255)
        v_gen = ImageDataGenerator(rescale=1.0/255)
        yt = y_train
        yc = y_val if y_val is not None else None
    else:
        if data_augmentation:
            t_gen = ImageDataGeneratorMO(rescale=1.0/255,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         brightness_range=[0.75, 1.25])
        else:
            t_gen = ImageDataGeneratorMO(rescale=1.0/255)
        v_gen = ImageDataGeneratorMO(rescale=1.0/255)
        # Split into list of class vectors per ImageDataGeneratorMO requirements
        yt = [np.array([x[i] for x in y_train]) for i in range(y_train.shape[1])]
        yc = None
        if y_val is not None:
            yc = [np.array([x[i] for x in y_val]) for i in range(y_val.shape[1])]

    # Train/validation split if not provided
    if X_val is None or (level == 1 and yc is None) or (level == 2 and yc is None):
        X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train,
                                                  shuffle=True,
                                                  test_size=AmfConfig.get('vfrac') / 100.0,
                                                  random_state=42)
        if level == 2:
            y_tr = [np.array([x[i] for x in y_tr]) for i in range(y_tr.shape[1])]
            y_va = [np.array([x[i] for x in y_va]) for i in range(y_va.shape[1])]
        yt, yc = y_tr, y_va
        xt, xc = X_tr, X_va
    else:
        xt, xc = X_train, X_val

    # Determine class weights
    weights = class_weights(yt if level == 1 else np.array(y_train)) if level == 1 else \
              {h: None for h in AmfConfig.get('header')}  # Keras multi-output weights would be dict; skipping here

    callbacks = get_callbacks()

    history = model.fit(t_gen.flow(xt, yt, batch_size=batch_size),
                        steps_per_epoch=max(1, len(xt) // batch_size),
                        class_weight=weights if level == 1 else None,
                        epochs=epochs,
                        validation_data=v_gen.flow(xc, yc, batch_size=batch_size),
                        validation_steps=max(1, len(xc) // batch_size),
                        callbacks=callbacks,
                        verbose=2)

    return model, history.history

# --- End modular training helpers ---


def import_settings(path):
    """
    Imports image settings stored in the auxiliary ZIP archive
    associated with the given image.

    :param path: Path to an input image.
    :return: Dictionary containing image settings
    :rtype: dict
    """

    zfile = get_zipfile(path)

    try:

        assert zf.is_zipfile(zfile)

        with zf.ZipFile(zfile, 'r') as z:

            assert AmfSave.IMG_SETTINGS in z.namelist()

            raw_text = z.read(AmfSave.IMG_SETTINGS).decode('utf-8')
            return yaml.safe_load(raw_text)

    except AssertionError:

        return {'tile_edge': AmfConfig.get('tile_edge')}



def import_annotations(path):
    """
    Imports tile annotations from the auxiliary ZIP archive
    associated with the given input image.

    :param path: Path to an input image.
    :return: Pandas dataframe containing annotations
    :rtype: pd.DataFrame
    """

    zfile = get_zipfile(path)

    try:

        assert zf.is_zipfile(zfile)

        with zf.ZipFile(zfile, 'r') as z:

            base = 'col' if AmfConfig.get('level') == 1 else 'myc'
            table = f'{base}.tsv'

            assert table in z.namelist()

            raw_data = z.read(table).decode('utf-8')
            return pd.read_csv(io.StringIO(raw_data), sep='\t')

    except AssertionError:

        return None



def estimate_background_subsampling(input_dataset):
    """
    Calculates the percentage of background tiles to omit
    to ensure near equal representation of all annotation classes
    in the training dataset. Background tiles are more abundant
    than roots in most images.

    :param input_dataset: Input dataset containing annotations.
    :return: The percentage of background tiles to omit.
    :rtype: int
    """

    # The concept of background tiles does not apply to CNN2.
    if AmfConfig.get('level') == 2 or AmfConfig.get('drop') == 0:

        return 0

    # Counts annotations for each input table.
    count_list = [x[2].sum() for x in input_dataset]

    # Generates the grand total.
    counts = functools.reduce(operator.add, count_list)

    # Number of tiles annotated as 'background/not a root/other'.
    background = counts['X']

    # Other tiles (i.e. annotated as 'colonised' or 'non-colonised' root).
    other_counts = counts.drop(['row', 'col', 'X'])

    # Average tile count per annotation class.
    average = round(sum(other_counts.values) / len(other_counts))

    # Excess background tiles compared to other classes.
    excess = background - average

    if excess <= 0:

        return 0

    else:

        x = round(excess * 100 / background)
        AmfLog.info(f'{x}% of background tiles will be ignored', indent=1)
        return x



def print_table_header():
    """
    Print table header.
    """
    if AmfConfig.get('level') == 1:

        print('Filename\tImage size (px)\tTile size (px)\tM+\tM-\tX\tDiscarded')

    else:

        print('Filename\tImage size (px)\tTile size (px)\tA\tV\tH\tI')



def print_image_stats(path, image, config, annots, discarded):
    """
    Print image information.
    """
    
    edge = config['tile_edge']
    base = os.path.basename(path)
    w = image.width
    h = image.height

    if AmfConfig.get('level') == 1:

        y = annots['Y'].sum()
        n = annots['N'].sum()
        x = annots['X'].sum()

        print(f'{base}\t{w}x{h}\t{edge}\t{y}\t{n}\t{x}\t{discarded}', flush=True)

    else:
    
        a = annots['A'].sum()
        v = annots['V'].sum()
        hy = annots['H'].sum()
        i = annots['I'].sum()
    
        print(f'{base}\t{w}x{h}\t{edge}\t{a}\t{v}\t{hy}\t{i}', flush=True)



def load_dataset(input_files):
    """
    Loads training tile set and their corresponding annotations.

    :param input_files: List of input images to use for training.
    :return: Numpy arrays containing tiles and one-hot encoded annotations.
    :rtype: tuple
    """

    print(f'[{AmfConfig.invite()}] Tile extraction.')

    # Load image settings and annotations.
    settings = [import_settings(path) for path in input_files]
    annotations = [import_annotations(path) for path in input_files]

    # Remove images without annotations.
    dataset = zip(input_files, settings, annotations)
    filtered_dataset = [x for x in dataset if x[2] is not None]

    # Terminate if there is no data to process.
    if len(filtered_dataset) == 0:

        AmfLog.error('Input images do not contain tile annotations. '
                     'Use amfbrowser to annotate tiles before training',
                     AmfLog.ERR_NO_DATA)

    # Determine the required amount of background subsampling (if active).
    subsampling = estimate_background_subsampling(filtered_dataset)

    tiles = []
    hot_labels = []
    header = AmfConfig.get('header')

    print_table_header()

    for path, config, annots in filtered_dataset:

        edge = config['tile_edge']
        AmfConfig.set('tile_edge', edge)

        # FIXME: Random access is inefficient. To achieve better
        # efficiency we would have to load tiles row by row.
        image = AmfSegm.load(path)
        
        # Extract tile sets (= original tile and augmented versions).
        # Repeat one-hot encoded annotations for each tile.
        discarded = 0
        for annot in annots.itertuples():

            if AmfConfig.get('level') == 1 and subsampling > 0 and \
               annot.X == 1 and random.uniform(0, 100) < subsampling:

                discarded += 1
                pass

            else:

                tile = AmfSegm.tile(image, annot.row, annot.col)
                tiles.append(tile)
                hot_labels.append(list(annot[3:]))

        print_image_stats(path, image, config, annots, discarded)

        del image

    return np.array(tiles, np.float32), np.array(hot_labels, np.uint8)



def class_weights(one_hot_labels):
    """
    Computes weights to counteract class imbalance and
    display statistics. Note: this functions requires
    Tensorflow 2.1 (and Keras 2.3.1). A bug in TF makes it
    impossible to use class_weights to models with multiple
    outputs. This bug is active on January 2021.
    Reference: https://github.com/tensorflow/tensorflow/issues/40457

    :param one_hot_labels: Hot labels encoding tile annotations.
    :return: Dictionary of class weights.
    :rtype: dict
    """

    print(f'[{AmfConfig.invite()}] Class weights')

    if AmfConfig.colonization():

        # For instance, [[0, 0, 1], [1, 0 , 0]] returns [2, 0]
        hot_indexes = np.argmax(one_hot_labels, axis=1)
        unique = np.unique(hot_indexes)
        class_weights = compute_class_weight('balanced',
                                             classes=unique,
                                             y=hot_indexes)

        # Annotation classes that do not ocurr in the input dataset.
        missing = [x for x in [0, 1, 2] if x not in unique]

        if missing != []:
            all_indices = list(unique) + missing
            all_weights = list(class_weights) + [0] * len(missing)
            class_weights = sorted(list(zip(all_indices, all_weights)))
            class_weights = [y for _, y in class_weights]

        for cls, num, w  in zip(AmfConfig.get('header'),
                                np.bincount(hot_indexes),
                                class_weights):

            frac = int(round(100.0 * num / len(one_hot_labels)))
            print(f'    - Class {cls}: {num} tiles ({frac}% of total).')
            print(f'    - Training weight: %.2f' % (w))

        return dict(enumerate(class_weights))

    else:

        class_weights = [compute_class_weight('balanced',
                                              classes=np.unique(y),
                                              y=y) for y in one_hot_labels]

        sums = [np.bincount(x) for x in one_hot_labels]
        for cls, ws, sums in zip(AmfConfig.get('header'), class_weights, sums):

            print('    - ConvNet %s: %d active (weight: %.2f), '
                  '%d inactive (weight: %.2f).' % (cls, sums[1], ws[1],
                                                   sums[0], ws[0]))

        # Output format: {'A': {0: wA0, 1: wA1}, 'V': {0: wV0, 1:wV1}, ...}
        # where wA0, w1A, wV0, and wV1 are weights (cf. compute_class_weight).
        return {x: dict(enumerate(y)) for x, y in zip(AmfConfig.get('header'),
                                                      class_weights)}



def get_callbacks():
    """
    Configures Keras callbacks to enable early stopping
    and learning rate reduction when reaching a plateau.

    :return: List of callback monitors.
    :rtype: list
    """

    # Prevent overfitting and restores best weights.
    e = EarlyStopping(monitor='val_loss',
           min_delta=0,
           patience=AmfConfig.get('patience'),
           verbose=1,
           mode='auto',
           restore_best_weights=True)
    AmfConfig.set('early_stopping', e)

    # Reduce learning rate for fine tuning.
    r = ReduceLROnPlateau(monitor='val_loss',
           factor=0.2,
           patience=2,
           verbose=1,
           min_lr=0.000001)
    AmfConfig.set('reduce_lr_on_plateau', r)

    return [e, r]



class ImageDataGeneratorMO(ImageDataGenerator):
    """
    Patched version of Keras's ImageDataGenerator to support multiple
    single-variable outputs. This requires output data to be a list of
    identical-length 1D NumPy arrays. Any inefficiency is negligible.
    Reference: https://github.com/keras-team/keras/issues/3761
    """

    def flow(self, x, y=None, **kwargs):
        if y is None or \
        not isinstance(y,list) or \
        any([yy.ndim != 1 for yy in y]) or \
        any([yy.size != len(x) for yy in y]):
           raise ValueError('ImageDataGeneratorMO requires a '
                            'list of outputs, each a 1D NumPy array of the '
                             'same length as the input.')

        y_sing = np.transpose(np.asarray(y))

        generator_sing = super(ImageDataGeneratorMO,
                               self).flow(x, y_sing, **kwargs)

        while True:
            batch_x, batch_y_sing = next(generator_sing)
            batch_y = [yy for yy in np.transpose(batch_y_sing)]
            yield batch_x, batch_y



def save_model_architecture(model):
    """
    Saves neural network architecture, parameters count, etc.

    :param model: Model to save.
    """

    if AmfConfig.get('summary'):

        cnn = 'CNN%d' % (AmfConfig.get('level'))
        path = os.path.join(AmfConfig.get('outdir'), f'{cnn}_summary.txt')

        with open(path, 'w') as sf:

            with redirect_stdout(sf):

                model.summary()

        path = os.path.join(AmfConfig.get('outdir'), f'{cnn}_architecture.png')
        keras.utils.plot_model(model, path, show_shapes=True)



def print_memory_usage():
    """
    Prints the amount of memory used to load training data.
    """

    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / (1024 * 1024)
    print(f"* Total memory used: {mb} Mb.")



img_index = 0

def data_augm(tile):
    """
    Two-step random tile augmentation based on blur and anticlockwise 90-degree
    rotation (step 1), and desaturation, inversion, chroma/hue alteration, or
    histogram equalisation (step 2). Random rotations and zoom are not used
    due to fungal structures occurring on edges in some tiles.

    :param tile: The tile to augment.
    :return: The augmented tile.
    :rtype: pyvips.Image
    """

    global img_index
    num = random.randint(1, 4)

    if num == 1:

        output = AmfImage.invert(tile)

    elif num == 2:

        output = AmfImage.grayscale(tile)

    elif num == 3:

        output = AmfImage.rotate_colours(tile)

    else:

        output = tile

    # May randomly save some augmented tiles.
    if img_index < AmfConfig.get('save_augmented_tiles') \
    and bool(random.getrandbits(1)):

        im = Image.fromarray(output.astype(np.uint8))
        img_index += 1
        im.save(os.path.join(AmfConfig.get('outdir'),
                'tile_%06d.png' % (img_index)))

    return output



def run(input_files):
    """
    Creates or loads a convolutional neural network, and trains it
    with the annotated tiles extracted from input images.

    :param input_files: List of input images to train with.
    """

    # Input model (either new or pre-trained).
    model = AmfModel.load()

    # Save model information (layers and graph) upon user request.
    save_model_architecture(model)

    # Input tiles and their corresponding annotations.
    tiles, labels = load_dataset(input_files)

    print_memory_usage()

    # Generates training and validation datasets.
    xt, xc, yt, yc = train_test_split(tiles, labels,
                                      shuffle=True,
                                      test_size=AmfConfig.get('vfrac') / 100.0,
                                      random_state=42)

    t_gen = None
    v_gen = None

    if AmfConfig.get('level') == 1:

        # Root segmentation (colonized vs non-colonized vs background).
        # ConvNet I has a standard, single input/single output architecture,
        # and can use ImageDataGenerator.
        if AmfConfig.get('data_augm'):
            t_gen = ImageDataGenerator(rescale=1.0 / 255,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       brightness_range=[0.75, 1.25],
                                       preprocessing_function=data_augm)
        else:
            t_gen = ImageDataGenerator(rescale=1.0 / 255)

        v_gen = ImageDataGenerator(rescale=1.0 / 255)

    else:

        # AM fungal structures (arbuscules, vesicles, hyphae).
        # ConvNet II has multiple outputs. ImageDataGenerator is not suitable.
        # Reference: https://github.com/keras-team/keras/issues/3761
        if AmfConfig.get('data_augm'):
            t_gen = ImageDataGeneratorMO(rescale=1.0 / 255,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         brightness_range=[0.75, 1.25],
                                         preprocessing_function=data_augm)
        else:
            t_gen = ImageDataGeneratorMO(rescale=1.0 / 255)

        v_gen = ImageDataGeneratorMO(rescale=1.0 / 255)

        # Reshape one-hot data in a way suitable for ImageDataGeneratorMO:
        # [[a1 v1 h1]...[aN vN hN]] -> [[a1...aN] [v1...vN] [h1...hN]]
        nclasses = len(AmfConfig.get('header')) # (= 4)
        yt = [np.array([x[i] for x in yt]) for i in range(nclasses)]
        yc = [np.array([x[i] for x in yc]) for i in range(nclasses)]

    # Determine weights to counteract class imbalance.
    weights = class_weights(yt)

    bs = AmfConfig.get('batch_size')

    his = model.fit(t_gen.flow(xt, yt, batch_size=bs),
                    steps_per_epoch=len(xt) // bs,
                    class_weight=weights,
                    epochs=AmfConfig.get('epochs'),
                    validation_data=v_gen.flow(xc, yc, batch_size=bs),
                    validation_steps=len(xc) // bs,
                    callbacks=get_callbacks(),
                    verbose=2)

    AmfSave.training_data(his.history, model)
