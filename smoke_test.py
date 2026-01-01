# Smoke test for AMFinder environment
import sys
import os
print('Python:', sys.version)
try:
    import tensorflow as tf
    print('TensorFlow:', tf.__version__)
    from tensorflow import keras
    print('tf.keras OK')
except Exception as e:
    print('TensorFlow import failed:', e)

# Try to import project module
repo_dir = os.path.dirname(__file__)
sys.path.insert(0, repo_dir)
sys.path.insert(0, os.path.join(repo_dir, 'amf'))
try:
    import amfinder_model as AmfModel
    import amfinder_config as AmfConfig
    print('Imported amfinder_* modules')
    AmfConfig.set('level', 1)
    AmfConfig.set('learning_rate', 0.001)
    m = AmfModel.create_cnn1()
    print('create_cnn1() ->', type(m))
except Exception as e:
    print('Project import or model creation failed:', e)
