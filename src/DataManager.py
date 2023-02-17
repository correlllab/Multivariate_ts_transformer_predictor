import os, sys
import glob
import pandas as pd
import numpy as np
import tensorflow
import csv
from dateutil.parser import parse
from YamlLoader import YamlLoader
import random
from sklearn.preprocessing import RobustScaler
from copy import deepcopy
import matplotlib.pyplot as plt


class DataManager:
    def __init__(self, main_path: str, config: dict) -> None:
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

        in_dir_exists = self._ensure_dir_exists(dir=in_dir, dir_type='read')
        out_dir_exists = self._ensure_dir_exists(dir=out_dir, dir_type='write')

        # Start by removing all files in out_dir (avoid duplicates!)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

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
    # '*.npy' is the default extension due to speed advantages (also reading from csv can derive in unexpected behavior if
    # csv files not treated properly first, i.e. need to be modified)
    def load_files_to_np_array(self, dir: str, extension: str = '*.npy', shuffle: bool = False, verbose: bool = False):
        if extension not in ['*.npy', '*.csv']:
            if verbose:
                print('File extension not supported')
            return []
        
        files = glob.glob(dir + extension)
        if verbose:
            print(f'Found {len(files)} {extension} files')

        if shuffle:
            random.shuffle(files)

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


    def run_pipeline(self, files_dir: str, in_dir: str = '', out_dir: str = '', file_extension: str = '*.npy', update_files: bool = False, verbose: bool = False):
        files = glob.glob(files_dir + file_extension)

        # Check if files of the extension type exist in dir
        # If files do not exist, create them
        failed = []
        if not files or update_files:
            if file_extension == '*.npy':
                failed = self.create_npy_files(in_dir=in_dir, out_dir=out_dir, ext_name=file_extension)

        # Load files data into np array
        data = self.load_files_to_np_array(dir=files_dir, extension=file_extension)

        # Apply standarization only to F-T columns (i.e. rows 1:6 all included, with 1:7 np excludes 7 so its 1 to 6)
        st_data = deepcopy(data)
        for index, d in enumerate(st_data):
            transformer = RobustScaler().fit(d[:, 1:7])
            st_data[index][:, 1:7] = transformer.transform(d[:, 1:7])

        # Transpose and add zero padding
        pad_data = deepcopy(st_data)
        max_len = np.max([d.shape[0] for d in pad_data])
        for index, d in enumerate(pad_data):
            pad_data[index] = np.pad(d.transpose(), ((0, 0), (0, max_len - d.shape[0])), mode='constant')

        return failed, data, st_data, np.array(pad_data)




def make_plot(data, st_data, pad_data):
    d = data[8]
    fx = d[:, 1]
    fy = d[:, 2]
    fz = d[:, 3]
    tx = d[:, 4]
    ty = d[:, 5]
    tz = d[:, 6]

    d = st_data[8]
    st_fx = d[:, 1]
    st_fy = d[:, 2]
    st_fz = d[:, 3]
    st_tx = d[:, 4]
    st_ty = d[:, 5]
    st_tz = d[:, 6]

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    axs[0, 0].plot(fx)
    axs[0, 0].plot(fy)
    axs[0, 0].plot(fz)
    axs[0, 0].set_title('Force')

    axs[0, 1].plot(tx)
    axs[0, 1].plot(ty)
    axs[0, 1].plot(tz)
    axs[0, 1].set_title('Torque')

    axs[1, 0].plot(st_fx)
    axs[1, 0].plot(st_fy)
    axs[1, 0].plot(st_fz)
    axs[1, 0].set_title('Standarized force')

    axs[1, 1].plot(st_tx)
    axs[1, 1].plot(st_ty)
    axs[1, 1].plot(st_tz)
    axs[1, 1].set_title('Standarized torque')

    d = pad_data[8]
    st_fx = d[1, :]
    st_fy = d[2, :]
    st_fz = d[3, :]
    st_tx = d[4, :]
    st_ty = d[5, :]
    st_tz = d[6, :]

    axs[2, 0].plot(st_fx)
    axs[2, 0].plot(st_fy)
    axs[2, 0].plot(st_fz)
    axs[2, 0].set_title('Standarized force with padding')

    axs[2, 1].plot(st_tx)
    axs[2, 1].plot(st_ty)
    axs[2, 1].plot(st_tz)
    axs[2, 1].set_title('Standarized torque with padding')

    plt.show()


if __name__ == '__main__':
    # System and Tensorflow checks ----------------------------------------------------------------
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
    # ---------------------------------------------------------------------------------------------
    
    MAIN_PATH = os.path.dirname(os.path.dirname(__file__))
    SRC_PATH = os.path.dirname(os.path.realpath(__file__))

    yl = YamlLoader()
    conf = yl.load_yaml(path='src/config/data_config.yaml')
    """ with open('src/config/data_config.yaml', 'r') as config_file:
        try:
            conf = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            print('ERROR:' + e) """


    dm = DataManager(main_path=MAIN_PATH, config=conf)
    if conf['create_npys']:
        for target in conf['targets']:
            failed = dm.create_npy_files(
                        in_dir=MAIN_PATH+conf['dirs'][target],
                        out_dir=MAIN_PATH+conf['dirs']['npys']+target+'/',
                        ext_name=target
                    )
            print(f'The following .npy files failed to create for target {target}: \n{failed}')


    data_dict = {k: None for k in conf['targets']}
    for target in conf['targets']:
        failed, data, st_data, pad_data = dm.run_pipeline(
            files_dir=MAIN_PATH + conf['dirs']['npys'] + target + '/'
        )

        data_dict[target] = {
            'failed': failed,
            'data': data,
            'st_data': st_data,
            'pad_data': pad_data
        }

        # print(f'The following files failed to load:\n{failed}')

        # plot = False

        # if plot:
        #     make_plot(data, st_data, pad_data)

        print(failed, data, st_data, pad_data)