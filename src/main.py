import os, sys
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import yaml

gpus = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
print( f"Found {len(gpus)} GPUs!" )
for i in range( len( gpus ) ):
    try:
        tensorflow.config.experimental.set_memory_growth(device=gpus[i], enable=True)
        tensorflow.config.experimental.VirtualDeviceConfiguration( memory_limit = 1024*3 )
        print( f"\t{tensorflow.config.experimental.get_device_details( device=gpus[i] )}" )
    except RuntimeError as e:
        print( '\n', e, '\n' )

devices = tensorflow.config.list_physical_devices()
print( "Tensorflow sees the following devices:" )
for dev in devices:
    print( f"\t{dev}" )

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
MAIN_PATH = os.path.dirname(os.path.dirname(__file__))
print(MAIN_PATH)