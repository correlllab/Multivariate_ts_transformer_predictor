import os, sys
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import yaml
import csv
from dateutil.parser import parse

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


class DataManager:
    def __init__(self, main_path: str, config: dict):
        self.main_path = main_path
        self.preemptive = self.main_path + config['dirs']['preemptive']
        self.reactive = self.main_path + config['dirs']['reactive']
        self.training = self.main_path + config['dirs']['training']
        self.csvs = self.main_path + config['dirs']['modified_csvs']
        self.npys = self.main_path + config['dirs']['npys']


    def modify_csv_files(self):
        pass


    # from James' https://github.com/correlllab/Efficiency_from_Failure_Classification/blob/master/RAL2022/data_processing.py
    def _csv_to_npy(self, input_file: str, out_dir: str, extended_name: str = '', time_col: int = 0, verbose: bool = False):
        """ Convert CSV directly to NPY while only changing timecodes """

        if verbose:
            print("Converting" , input_file , ", Exists?:" , os.path.isfile(input_file) , end = ", Begin ... ")

        t_0 = None
        arrLst = []
        
        try:
            # 1. Attempt to read the CSV
            with open(input_file) as f:
                csvReader = csv.reader(f)
                for i, row in enumerate(csvReader):
                    # 2. Convert the timecode column
                    t = parse(row[0])
                    if i == 0:
                        t_0  = t
                    t_ms = float((t - t_0).total_seconds()*1000)
                    rowLst = [t_ms, ]
                    for item in row[time_col + 1: ]:
                        elem = eval(item)
                        if type(elem) in (list, tuple):
                            rowLst.extend(elem)
                        else:
                            rowLst.append(elem)
                    arrLst.append(rowLst)

            # 3. Load matrix
            savMatx = np.array(arrLst)

            # N. Save NPY
            filename = os.path.split(input_file)[-1]
            newNPYpath = os.path.join(out_dir , filename.split('.')[0] + '_' + extended_name + '.npy') 
            np.save(newNPYpath, savMatx, allow_pickle = 0)

            if verbose:
                print("SUCCESS!")

            return newNPYpath

        except Exception as err:
            if verbose:
                print(f"FAILURE! with error: {err}")

            return None


    def _ensure_dir_exists(self, dir: str, dir_type: str, verbose: bool = False):
        """
        Check whether 'dir' exists, if it does not exist:
            - if 'dir_type' == 'read' (i.e. input dir) ==> return error
            - if 'dir_type' == 'write' (i.e. output dir) ==> create it
        """

        exists = os.path.isdir(dir)
        if not exists and dir_type == 'write':
            os.makedirs(dir)
            exists = os.path.isdir(dir)

        return exists


    def create_npy_files(self, in_dir: str, out_dir: str, ext_name: str, time_col: int = 0, verbose: bool = False):
        """ Create npy files from csvs in 'in_dir' and saves them in 'out_dir' """

        in_dir_exists = self._ensure_dir_exists(dir=in_dir, dir_type='read')
        out_dir_exists = self._ensure_dir_exists(dir=out_dir, dir_type='write')

        if verbose:
            print("The input  directory is" , in_dir  , "Exists?:" , in_dir_exists)
            print("The output directory is" , out_dir , "Exists?:" , out_dir_exists)

        if (in_dir_exists and out_dir_exists):
            # TODO: create npy files from csvs inside 'in_dir'
            pass







if __name__ == '__main__':
    pass
