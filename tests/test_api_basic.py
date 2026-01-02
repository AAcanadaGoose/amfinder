import os
import sys
import numpy as np
from PIL import Image

# Make amf/ importable
REPO = os.path.dirname(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(REPO, 'amf'))

from amfinder_api import AMFinder  # noqa: E402


def main():
    # Initialize API
    amf = AMFinder(tile_size=126, batch_size=8)

    # Load CNN1 model (bundled under amf/trained_networks)
    amf.load_model('CNN1v2.h5', level=1)

    img_path = os.path.join(REPO, 'Sample_photos', 'clear1.JPG')

    # 1) Predict from file path
    df = amf.predict(img_path, level=1, return_format='dataframe')
    print('Prediction DF shape:', df.shape)
    print(df.head())

    # 2) Predict from numpy array
    arr = np.array(Image.open(img_path).convert('RGB'))
    df2 = amf.predict(arr, level=1)
    print('Array prediction equal row count:', len(df2) == len(df))

    # 3) Extract tiles and predict on tiles
    tiles, nrows, ncols = amf.extract_tiles(img_path)
    preds = amf.predict_tiles(tiles, level=1)
    print('Tiles:', tiles.shape, 'Preds:', preds.shape, 'Grid:', (nrows, ncols))

    # 4) Convert probabilities to one-hot annotations
    ann = amf.convert(df, level=1)
    print('Converted annotations:')
    print(ann.head())


if __name__ == '__main__':
    main()
