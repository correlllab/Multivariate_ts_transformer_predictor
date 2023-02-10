import os, sys
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import yaml
import csv
from dateutil.parser import parse


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


    # adapted from James' 'csv_to_npy_dir' function in https://github.com/correlllab/Efficiency_from_Failure_Classification/blob/83220b03ef4c41ebe2a57eb1a3324a75da8c8426/RAL2022/data_processing.py#L75
    def create_npy_files(self, in_dir: str, out_dir: str, ext_name: str = '', time_col: int = 0, verbose: bool = False):
        """ Create npy files from csvs in 'in_dir' and saves them in 'out_dir' """

        # Start by removing all files in out_dir (avoid duplicates!)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

        in_dir_exists = self._ensure_dir_exists(dir=in_dir, dir_type='read')
        out_dir_exists = self._ensure_dir_exists(dir=out_dir, dir_type='write')

        if verbose:
            print("The input  directory is" , in_dir  , "Exists?:" , in_dir_exists)
            print("The output directory is" , out_dir , "Exists?:" , out_dir_exists)

        if (in_dir_exists and out_dir_exists):
            # TODO: create npy files from csvs inside 'in_dir'
            files = os.listdir(in_dir)
            if verbose:
                print( "There are" , len(files) , "files to process in this directory\n" )

            failed = []
            for f in files:
                full_path = in_dir + f
                if verbose:
                    print(f'Processing {full_path} ...')
                
                try:
                    _ = self._csv_to_npy(input_file=full_path, out_dir=out_dir, extended_name=ext_name)
                except Exception as e:
                    failed.append(f)
                    if verbose:
                        print(f'Conversion failed for {f} with error: {e}')
        elif not in_dir_exists and verbose:
            print('in_dir does not exist')
        elif not out_dir_exists and verbose:
            print('out_dir does not exist and could not be created')

        return failed

    # from James' notebook https://github.com/correlllab/Efficiency_from_Failure_Classification/blob/master/RAL2022/00_FCN-1.ipynb section 'Load Data'
    def load_files_to_np_array(self, dir: str, extension: str, shufle: bool = False, verbose: bool = False):
        if extension not in ['*.npy', '*.csv']:
            if verbose:
                print('File extension not supported')
            return []
        
        files = glob.glob(dir + extension)
        if verbose:
            print(f'Found {len(files)} {extension} files')

        if shufle:
            shufle(files)

        data = []
        succ = 0
        fail = 0
        for f in files:
            if extension == '*.npy':
                mat = np.array(np.load(f)).astype(dtype=float)
            elif extension == '*.csv':
                # this will only work with the csvs inside the modified csv folder
                mat = np.array(np.loadtxt(f, skiprows=1, delimiter=',')).astype(dtype=float)
            if mat[0, 7] == 1:
                succ += 1
            elif mat[0, 7] == 0:
                fail += 1
            data.append(mat)

        if verbose and data:
            print(f'Created {len(data)} episode matrices')
            print(f'Successes = {succ} | Failures = {fail}')
            print(f'Success rate = {succ/(succ+fail)} | Failure rate = {fail/(succ+fail)}')
        
        return data




if __name__ == '__main__':
    print( sys.version )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
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
    
    MAIN_PATH = os.path.dirname(os.path.dirname(__file__))
    SRC_PATH = os.path.dirname(os.path.realpath(__file__))

    with open('src/config/data_config.yaml', 'r') as config_file:
        try:
            conf = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            print('ERROR:' + e)


    dm = DataManager(main_path=MAIN_PATH, config=conf)
    """ failed = dm.create_npy_files(
                in_dir=MAIN_PATH+conf['dirs'][conf['target']],
                out_dir=MAIN_PATH+conf['dirs']['npys'],
                ext_name=conf['target']
             )
    print(f'The following .npy files failed to create: {failed}') """
    d = dm.load_files_to_np_array(
        dir=MAIN_PATH+conf['dirs']['npys'],
        extension='*.npy',
        shufle=conf['shuffle'],
        verbose=True
    )
    print(d)