import sys, os, glob
import random
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import tensorflow as tf

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import RobustScaler

from random import shuffle
from copy import deepcopy
from utilities.helper_functions import position_encode

# SHUFFLING ONLY ON EPISODES!!!! NOT ON WINDOWS

class DataPreprocessing:
    def __init__(self, sampling: str = 'over' or 'under', data: str = 'preemptive' or 'reactive' or 'training') -> None:
        self.sampling = sampling
        self.data_name = data
        self.datadir = os.path.join(os.path.dirname(os.path.abspath('../')), f'data/Npy_files/{data}/')
        self.shuffle = True
        self.data = None
        self.truncData = None
        self.window_data = None
        self.testFrac = 0.10
        self.N_ep = 0
        self.N_test = 0
        self.N_train = 0
        self.trainWindows = 0
        self.testWindows = 0
        self.X_train = None
        self.Y_train = None
        self.X_train_sampled = None
        self.Y_train_sampled = None
        self.X_train_enc = None
        self.X_test = None
        self.Y_test = None
        self.X_test_enc = None
        self.X_winTest = None
        self.Y_winTest = None
        self.rollWinWidth = None


    def load_data(self, verbose=False):
        """ Load 100 files so that there is balance between classes """
        ext      = "*.npy"
        print(self.datadir)
        npyFiles = glob.glob(self.datadir + ext)
        if verbose:
            print( f"Found {len(npyFiles)} {ext} files!" )

        if self.shuffle:
            shuffle( npyFiles )
            if verbose:
                print( "Shuffled files!" )

        epData = []
        N_s    = 0
        N_f    = 0
        for file in npyFiles:
            epMatx = np.array( np.load( file ) ).astype( dtype = float )
            # print( epMatx[0,7] )
            if epMatx[0,7] == 1.0:
                N_s += 1
            elif epMatx[0,7] == 0.0:
                N_f += 1
            epData.append( epMatx )
            if verbose:
                print( '>', end=' ' )
        
        if verbose:
            print( f"\nCreated {len(epData)} episode matrices!" )
            print( f"{N_s} successes, {N_f} failures, Success Rate: {N_s/(N_s+N_f)}, Failure Rate: {N_f/(N_s+N_f)}" )
        self.data = epData


    def set_episode_beginning(self, verbose=False):
        # Begin each ep at 1st imapct 
        truncData   = []
        winWidth    = 10
        FzCol       =  3
        spikeThresh = 0.05

        # For every episode
        for j, epMatx in enumerate( self.data ):
            # Look for the first spike in F_z
            N          = epMatx.shape[0]
            # if verbose:
            #     print( f"{j}\n{epMatx.shape} input data shape" )
            chopDex    = 0
            for bgn in range( int(1.5*50), N-winWidth+1 ):
                end     = bgn+winWidth
                FzSlice = epMatx[ bgn:end, FzCol ].flatten()
                # print( FzSlice.shape )
                # print( np.amax( FzSlice ), np.amin( FzSlice ), type( np.amax( FzSlice ) - np.amin( FzSlice ) ) )
                if np.abs( np.amax( FzSlice ) - np.amin( FzSlice ) ) >= spikeThresh:
                    # print( np.amax( FzSlice ), np.amin( FzSlice ) )
                    # print( FzSlice )
                    chopDex = bgn
                    # if verbose:
                    #     print( f"Relevant data at {chopDex*20/1000.0} seconds!" )
                    break
            if (chopDex*20/1000) < 15.0:
                truncData.append( epMatx[ chopDex:,: ] )
            # else dump an episode that does not fit criteria, 2022-08-31: Dumped 5 episodes
            if verbose:
                print( '>', end=' ' )
        
        if verbose:
            print( f"\nTruncated {len(truncData)} episodes!" )
        self.truncData = truncData


    def get_complete_twist_windows(self, verbose=False):
        self.rollWinWidth = int(7.0 * 50) #int(8.5 * 50)
        windowData   = []
        for j, epMatx in enumerate( self.truncData ):
            R = epMatx.shape[0]
            # C = epMatx.shape[1]
            L = R - self.rollWinWidth + 1
            epWindows = np.zeros( (L,self.rollWinWidth,7,) )
            for i in range(L):
                end = i+self.rollWinWidth
                epWindows[i,:,:] = epMatx[ i:end, 1:8 ]
            windowData.append( epWindows )
            if verbose:
                print( f'{epWindows.shape}', end=' ' )
        if verbose:
            print( "\nDONE!" )
        self.window_data = windowData


    # def balance_window_data(self, verbose=False):
    #     window_limit = 100000
    #     positive_windows = []
    #     negative_windows = []
    #     positive_sum = 0.
    #     negative_sum = 0.
    #     positive_flag = False
    #     negative_flag = False
    #     stop_flag = False
    #     i = 0
    #     while not stop_flag:
    #         episode = self.window_data[i]
    #         if episode[0,0,6] == 1.0:
    #             if positive_sum <= window_limit:
    #                 positive_sum += episode.shape[0]
    #                 positive_windows.append(episode)
    #             else:
    #                 positive_flag = True
    #         elif episode[0,0,6] == 0.0:
    #             if negative_sum <= window_limit:
    #                 negative_sum += episode.shape[0]
    #                 negative_windows.append(episode)
    #             else:
    #                 negative_flag = True

    #         stop_flag = (positive_flag and negative_flag)
    #         i += 1

    #     if verbose:
    #         print(f'Truncated to {positive_sum} positive windows and {negative_sum} negative windows')

    #     # Concat the two lists and shuffle
    #     self.window_data = [*positive_windows, *negative_windows]
    #     shuffle(self.window_data)


    def stack_windows(self, verbose=False):
        self.testFrac     = 0.10
        self.N_ep         = len( self.window_data )
        self.N_test       = int(self.N_ep * self.testFrac)
        self.N_train      = self.N_ep - self.N_test
        self.trainWindows = 0
        self.testWindows  = 0

        i = 0

        for j in range( self.N_train ):
            ep = self.window_data[i]
            self.trainWindows += ep.shape[0]
            i += 1
            
        for j in range( self.N_test ):
            ep = self.window_data[i]
            self.testWindows += ep.shape[0]
            i += 1

        if verbose:
            print( f"{self.trainWindows} windows to Train and {self.testWindows} to Test" )
            print( f"All episodes accounted for?: {i == self.N_ep}, {i}, {self.N_ep}" )

        datasetTrain = np.zeros( (self.trainWindows, self.window_data[0].shape[1], self.window_data[0].shape[2],) )
        datasetTest  = np.zeros( (self.testWindows , self.window_data[0].shape[1], self.window_data[0].shape[2],) )

        i = 0
        j = 0
        for _ in range( self.N_train ):
            ep = self.window_data[i]
            for window in ep:
                datasetTrain[j,:,:] = window
                j += 1
            i += 1
            if verbose:
                print( "T", end = " " )
            
        j = 0
            
        for _ in range( self.N_test ):
            ep = self.window_data[i]
            for window in ep:
                datasetTest[j,:,:] = window
                j += 1
            i += 1
            if verbose:
                print( "V", end = " " )
                
                
        self.X_train = datasetTrain[ :, :, 0:6 ]
        self.Y_train = np.zeros( (self.trainWindows, 1) )
        if verbose:
            print( f"\nTrain X shape: {self.X_train.shape}" )
            print( f"\nTrain Y shape: {self.Y_train.shape}" )

        self.X_test = datasetTest[ :, :, 0:6 ]
        self.Y_test = np.zeros( (self.testWindows, 1) )
        if verbose:
            print( f"\nTest X shape: {self.X_test.shape}" )
            print( f"\nTest Y shape: {self.Y_test.shape}" )

        i   = 0 
        k   = 0
        pos = 0
        neg = 0

        for j in range( self.N_train ):
            ep = self.window_data[i]
            for window in ep:
                if datasetTrain[k,0,6] == 1.0:
                    self.Y_train[k,0] = 0
                    pos += 1
                elif datasetTrain[k,0,6] == 0.0:
                    self.Y_train[k,0] = 1
                    neg += 1
                else:
                    raise ValueError( "BAD LABEL" )
                k += 1
            i += 1
            if verbose:
                print( i, end = " " )
            
        k = 0

        for j in range( self.N_test ):
            ep = self.window_data[i]
            # print()
            for window in ep:
                # print(datasetTest[k,0,6], end=' ')
                # print( datasetTest[k,-1,:] )
                if datasetTest[k,0,6] == 1.0:
                    self.Y_test[k,0] = 0
                    pos += 1
                elif datasetTest[k,0,6] == 0.0:
                    self.Y_test[k,0] = 1
                    neg += 1
                else:
                    raise ValueError( "BAD LABEL" )
                k += 1
            i += 1
            if verbose:
                print( i, end = " " )
        
        if verbose:
            print( '\n' )
            print( self.Y_train.shape, self.Y_test.shape )
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, num_classes=2)
        self.Y_test  = tf.keras.utils.to_categorical(self.Y_test, num_classes=2)
        if verbose:
            print( self.Y_train.shape, self.Y_test.shape )
            print( f"\nThere are {pos} ({(pos * 100) / (pos + neg):.2f}%) passing windows and {neg} ({(neg * 100) / (pos + neg):.2f}%) failing windows!, Total: {pos+neg}" )


    def capture_test_episodes(self, verbose=False):
        self.X_winTest = []
        self.Y_winTest = []

        for i in range( self.N_train, self.N_ep ):
            ep = self.window_data[i]
            self.X_winTest.append( ep[ :, :, 0:6 ] )
            y_i = np.zeros( (ep.shape[0], 2) )
            if ep[ 0, 0, 6 ] == 1.0:
                y_i[:,:] = [1.0, 0.0]
                self.Y_winTest.append( y_i ) 
            elif ep[ 0, 0, 6 ] == 0.0:
                y_i[:,:] = [0.0, 1.0]
                self.Y_winTest.append( y_i )
            if verbose:
                print( '>', end=' ' )

        self.X_winTest = np.asanyarray(self.X_winTest, dtype=object)
        self.Y_winTest = np.asanyarray(self.Y_winTest, dtype=object)

        if verbose:
            print( f"\nDONE! Captured {self.X_winTest.shape}/{self.Y_winTest.shape} TEST episodes." )


    def balance_classes(self, verbose=False):
        if verbose:
            print('    ====> CLASSES DISTRIBUTION BEFORE:')
            print(f'        Passes = {int(sum(self.Y_train[:,0]))}; Fails = {int(sum(self.Y_train[:,1]))}\n')

        if self.sampling == 'over':
            # TODO: try oversampler, probably does not work like this
            # oversampler = RandomOverSampler(sampling_strategy='minority')
            oversampler = SMOTE()
            self.X_train_sampled, self.Y_train_sampled = oversampler.fit_resample(self.X_train.reshape(self.X_train.shape[0], -1), self.Y_train)
            # oversampler.fit_resample(self.X_train[:,:,0], self.Y_train)
            # self.X_train_sampled = deepcopy(self.X_train[oversampler.sample_indices_])
            # self.Y_train_sampled = deepcopy(self.Y_train[oversampler.sample_indices_])
        elif self.sampling == 'under':
            undersampler = RandomUnderSampler(sampling_strategy='majority')
            undersampler.fit_resample(self.X_train[:,:,0], self.Y_train)
            self.X_train_sampled = deepcopy(self.X_train[undersampler.sample_indices_])
            self.Y_train_sampled = deepcopy(self.Y_train[undersampler.sample_indices_])
        else:
            self.X_train_sampled = deepcopy(self.X_train)
            self.Y_train_sampled = deepcopy(self.Y_train)

        if verbose:
            print('    ====> CLASSES DISTRIBUTION AFTER:')
            print(f'        Passes = {int(sum(self.Y_train_sampled[:,0]))}; Fails = {int(sum(self.Y_train_sampled[:,1]))}\n')
    #         print(self.X_train_sampled.shape, self.Y_train_sampled.shape)


    def scale_data(self, verbose=False):
        for index, ep in enumerate(self.data):
            transformer = RobustScaler().fit(ep[:, 1:7])
            self.data[index][:, 1:7] = transformer.transform(ep[:, 1:7])


    def run(self, save_data=False, verbose=False):
        if verbose:
            print('\n====> Loading data...\n')
        self.load_data(verbose=verbose)
        if verbose:
            print('\n====> Scaling data...\n')
        self.scale_data(verbose=verbose)
        if verbose:
            print('\n====> Setting episodes beginnings...\n')
        self.set_episode_beginning(verbose=verbose)
        if verbose:
            print('\n====> Getting complete twist windows...\n')
        self.get_complete_twist_windows(verbose=verbose)
        # if verbose:
        #     print('\n====> Balancing windows...\n')
        # self.balance_window_data(verbose=verbose)
        if verbose:
            print('\n====> Stacking windows...\n')
        self.stack_windows(verbose=verbose)
        if verbose:
            print('\n====> Capturing test episodes...\n')
        self.capture_test_episodes(verbose=verbose)
        if verbose:
            print('\n====> Balancing classes...\n')
        self.balance_classes(verbose=verbose)
        if verbose:
            print('\n====> Done preprocessing!\n')
            _ = input('Continue?:')
        if save_data:
            if verbose:
                print('\n====> Saving data into npy files...', end='\n')

            save_dir = f'../../data/data_manager/{self.data_name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(f'{save_dir}/{self.data_name}_X_train.npy', 'wb') as f:
                np.save(f, self.X_train, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_Y_train.npy', 'wb') as f:
                np.save(f, self.Y_train, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_X_train_sampled.npy', 'wb') as f:
                np.save(f, self.X_train_sampled, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_Y_train_sampled.npy', 'wb') as f:
                np.save(f, self.Y_train_sampled, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_X_test.npy', 'wb') as f:
                np.save(f, self.X_test, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_Y_test.npy', 'wb') as f:
                np.save(f, self.Y_test, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_X_winTest.npy', 'wb') as f:
                np.save(f, self.X_winTest, allow_pickle=True)

            with open(f'{save_dir}/{self.data_name}_Y_winTest.npy', 'wb') as f:
                np.save(f, self.Y_winTest, allow_pickle=True)

            if verbose:
                print('DONE\n')


if __name__ == '__main__':
    dp = DataPreprocessing(sampling='under', data='training')
    dp.run(save_data=False, verbose=True)
    print(type(dp.truncData))
    print(type(random.choice(dp.truncData)))
    print('ALL OK')
