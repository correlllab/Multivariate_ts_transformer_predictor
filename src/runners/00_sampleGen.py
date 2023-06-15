import numpy as np

DATA = ['reactive', 'training']
DATA_DIR = f'../../data/data_manager/{"_".join(DATA)}'

X_smpl = None

with open(f'{DATA_DIR}/{"_".join(DATA)}_X_test.npy', 'rb') as f:
    X_smpl = np.load( f, allow_pickle = True )

X_smpl = X_smpl[:64]

np.save( "OOPT_Data_Sample.npy", X_smpl, allow_pickle = True )