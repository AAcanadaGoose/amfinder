"""
Compatibility layer between AMFinder API and CLI.
Adapters to maintain backward compatibility.
"""

from typing import List

import amfinder_config as AmfConfig
from amfinder_api import AMFinder


def api_to_cli_config(amfinder_instance: AMFinder) -> dict:
    """
    Convert AMFinder instance config to CLI PAR dict.
    """
    return {
        'tile_edge': amfinder_instance.tile_size,
        'batch_size': amfinder_instance.batch_size,
        'super_resolution': bool(amfinder_instance.super_resolution),
    }


def cli_predict_wrapper(input_files: List[str]):
    """
    Wrapper for CLI predict using API backend.
    """
    amf = AMFinder(tile_size=AmfConfig.get('tile_edge'),
                   batch_size=AmfConfig.get('batch_size'),
                   super_resolution=AmfConfig.get('super_resolution'))
    level = AmfConfig.get('level')
    # Load model from configured path
    amf.load_model(AmfConfig.get('model'), level=level)
    # Run predictions and save via existing pipeline
    import amfinder_save as AmfSave
    for path in input_files:
        df = amf.predict(path, level=level, return_format='dataframe')
        # For compatibility, rely on original saver
        AmfSave.prediction_table(df, None, path)


def cli_train_wrapper(input_files: List[str]):
    """
    Wrapper for CLI train using API backend.
    """
    # CLI provides only images; annotations ZIPs are inferred by original code.
    # Here we defer to original train.run to avoid duplicating logic.
    import amfinder_train as AmfTrain
    AmfTrain.run(input_files)
