import os
import sys
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(REPO, 'amf'))

from amfinder_api import AMFinder  # noqa: E402


def main():
    amf = AMFinder(tile_size=126, batch_size=8)
    amf.load_model('CNN2v2.h5', level=2)  # bundled model filename

    img_path = os.path.join(REPO, 'Sample_photos', 'clear1.JPG')

    # Predict directly from numpy array
    arr = np.array(Image.open(img_path).convert('RGB'))
    df = amf.predict(arr, level=2, return_format='dataframe')
    print('CNN2 prediction DF shape:', df.shape)
    print(df.head())

    # Convert to one-hot annotations with threshold
    ann = amf.convert(df, level=2, threshold=0.5)
    print('CNN2 converted annotations:')
    print(ann.head())


if __name__ == '__main__':
    main()
