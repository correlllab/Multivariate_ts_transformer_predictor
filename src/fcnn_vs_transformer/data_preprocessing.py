
import sys, os, glob
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

class DataPreprocessing:
    def __init__(self) -> None:
        self.datadir = os.path.join(os.path.abspath("./"), "comp_data/")
        self.shuffle = True
        self.data = None
        self.truncData = None
        self.window_data = None
        self.testFrac = 0.20
        self.N_ep = 0
        self.N_test = 0
        self.N_train = 0
        self.trainWindows = 0
        self.testWindows = 0
        self.X_train = None
        self.Y_train = None
        self.X_train_under = None
        self.Y_train_under = None
        self.X_test = None
        self.Y_test = None
        self.X_winTest = None
        self.Y_winTest = None
        self.rollWinWidth = None


    def load_data(self):
        ext      = "*.npy"
        npyFiles = glob.glob(self.datadir + ext)
        print( f"Found {len(npyFiles)} {ext} files!" )

        if self.shuffle:
            shuffle( npyFiles )
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
            print( '>', end=' ' )
            
        print( f"\nCreated {len(epData)} episode matrices!" )
        print( f"{N_s} successes, {N_f} failures, Success Rate: {N_s/(N_s+N_f)}, Failure Rate: {N_f/(N_s+N_f)}" )
        self.data = epData


    def set_episode_beginning(self):
        # Begin each ep at 1st imapct 
        truncData   = []
        winWidth    = 10
        FzCol       =  3
        spikeThresh = 0.05
        vb          = 0

        # For every episode
        for j, epMatx in enumerate( self.data ):
            # Look for the first spike in F_z
            N          = epMatx.shape[0]
            if vb:
                print( f"{j}\n{epMatx.shape} input data shape" )
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
                    if vb:
                        print( f"Relevant data at {chopDex*20/1000.0} seconds!" )
                    break
            if (chopDex*20/1000) < 15.0:
                truncData.append( epMatx[ chopDex:,: ] )
            # else dump an episode that does not fit criteria, 2022-08-31: Dumped 5 episodes
            if vb:
                print( f"{ truncData[-1].shape } output data shape" )
                print()
            else:
                print( '>', end=' ' )
            
        print( f"\nTruncated {len(truncData)} episodes!" )
        self.truncData = truncData


    def get_complete_twist_windows(self):
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
            print( f'{epWindows.shape}', end=' ' )
        print( "\nDONE!" )
        self.window_data = windowData


    def stack_windows(self):
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
            print( "T", end = " " )
            
        j = 0
            
        for _ in range( self.N_test ):
            ep = self.window_data[i]
            for window in ep:
                datasetTest[j,:,:] = window
                j += 1
            i += 1
            print( "V", end = " " )
                
                
        self.X_train = datasetTrain[ :, :, 0:6 ]
        self.Y_train = np.zeros( (self.trainWindows, 1) )
        print( f"\nTrain X shape: {self.X_train.shape}" )
        print( f"\nTrain Y shape: {self.Y_train.shape}" )

        self.X_test = datasetTest[ :, :, 0:6 ]
        self.Y_test = np.zeros( (self.testWindows, 1) )
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
            print( i, end = " " )
            
        print( '\n' )
        print( self.Y_train.shape, self.Y_test.shape )
        self.Y_train = to_categorical( self.Y_train )
        self.Y_test  = to_categorical( self.Y_test  )
        print( self.Y_train.shape, self.Y_test.shape )
            
        print( f"\nThere are {pos} passing windows and {neg} failing windows!, Total: {pos+neg}" )


    def capture_test_episodes(self):
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
            print( '>', end=' ' )
            
        print( f"\nDONE! Captured {len(self.X_winTest)}/{len(self.Y_winTest)} TEST episodes." )


    def balance_classes(self):
        print('    ====> CLASSES DISTRIBUTION BEFORE:')
        print(f'        Passes = {int(sum(self.Y_train[:,0]))}; Fails = {int(sum(self.Y_train[:,1]))}\n')

        undersampler = RandomUnderSampler(sampling_strategy='majority')
        undersampler.fit_resample(self.X_train[:,:,0], self.Y_train)
        self.X_train_under = self.X_train[undersampler.sample_indices_]
        self.Y_train_under = self.Y_train[undersampler.sample_indices_]

        print('    ====> CLASSES DISTRIBUTION AFTER:')
        print(f'        Passes = {int(sum(self.Y_train_under[:,0]))}; Fails = {int(sum(self.Y_train_under[:,1]))}\n')
        print(self.X_train_under.shape, self.Y_train_under.shape)

    def run(self, verbose=False):
        if verbose:
            print('\n====> Loading data...\n')
        self.load_data()
        if verbose:
            print('\n====> Setting episodes beginnings...\n')
        self.set_episode_beginning()
        if verbose:
            print('\n====> Getting complete twist windows...\n')
        self.get_complete_twist_windows()
        if verbose:
            print('\n====> Stacking windows...\n')
        self.stack_windows()
        if verbose:
            print('\n====> Capturing test episodes...\n')
        self.capture_test_episodes()
        if verbose:
            print('\n====> Balancing classes...\n')
        self.balance_classes()
        if verbose:
            print('\n====> Done preprocessing!\n')
        _ = input('Continue?:')