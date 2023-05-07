########## INIT ###################################################################################

##### Imports #####
import pickle, os, sys, time
from time import sleep

import tensorflow


########## GPU / TENSORFLOW ########################################################################

def init_gpus_for_tf():
    """ Set up GPU(s) to support classification work in tensorflow-gpu """
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
        print



########## UTILITY CLASSES #########################################################################


class HeartRate:
    """ Sleeps for a time such that the period between calls to sleep results in a frequency <= 'Hz' """
    
    def __init__( self , Hz ):
        """ Create a rate object with a Do-Not-Exceed frequency in 'Hz' """
        self.period = 1.0 / Hz; # Set the period as the inverse of the frequency
        self.last = time.time()
    
    def sleep( self ):
        """ Sleep for a time so that the frequency is not exceeded """
        elapsed = time.time() - self.last
        if elapsed < self.period:
            time.sleep( self.period - elapsed )
        self.last = time.time()
        
        
class Periodic:
    """ Runs a task for as long as it is `running` """

    def __init__( self, runHz, daemon = 1 , pause_s = 0.10 ):
        """ Set flag """
        self.running  = 0
        self.rate     = HeartRate( runHz )  
        
    def repeat( self ):
        """ Repeat `task` until asked to stop """
        # NOTE: This function assumes that `task` is a member function that takes no args after `self`
        while self.running:
            self.task()
            self.rate.sleep()
        print( "\nTask stopped!" )
        
    def run( self, flag = 1 ):
        """ Set the running flag """
        self.running = flag

    def stop( self ):
        """ Ask that the execution stops """
        # 1. Ask the thread to stop
        self.running = 0 
        # 2. Wait for thread to notice
        sleep( _NETPAUSE_S )
        
    def task( self ):
        """ VIRTUAL PLACEHOLDER """
        raise NotImplementedError( "You must OVERRIDE `self.task`!" )
        
        
        
class Counter:
    """ Keeps tracks of calls """
    
    def __init__( self ):
        """ Set the counter to zero """
        self.count = 0
    
    def reset( self ):
        """ Set the counter to zero """
        self.__init__()
    
    def __call__( self ):
        """ Increment the counter """
        self.count += 1
        return self.count
        
    def __str__( self ):
        """ Return the count as a string """
        return str( self.count )
    
    def set_count( self , i ):
        """ Manually set the counter """
        self.count = int( i )
        
        
class CounterDict( dict ): 
    """ The counter object acts as a dict, but sets previously unused keys to 0 , in the style of CS 6300 @ U of Utah """

    def __init__( self , *args , **kw ):
        """ Standard dict init """
        dict.__init__( self , *args , **kw )
        if "default" in kw:
            self.defaultReturn = kw['default']
        else:
            self.defaultReturn = 0

    def set_default( self , val ):
        """ Set a new default value to return when there is no """
        self.defaultReturn = val

    def __getitem__( self , a ):
        """ Get the val with key , otherwise return 0 if key DNE """
        if a in self: 
            return dict.__getitem__( self , a )
        return self.defaultReturn
        
    def count( self, key ):
        """ Increment the value stored at the key as though instances of the key were being counted """
        if key in self: 
            self[ key ] += 1 
        else:
            self[ key ] = 1

    # __setitem__ provided by 'dict'

    def sorted_keyVals( self ):
        """ Return a list of sorted key-value tuples """
        sortedItems = self.items()
        sortedItems.sort( cmp = lambda keyVal1 , keyVal2 :  np.sign( keyVal2[1] - keyVal1[1] ) )
        return sortedItems
    
    def unroll_to_lists( self ):
        """ Return keys and values in associated pairs """
        rtnKeys = list( self.keys() )
        rtnVals = [ self[k] for k in rtnKeys ] # Positions must match, iterate over above list
        return rtnKeys , rtnVals
        

########## Model Save/Load ########################################################################


def pkl_obj( fName, obj, binary = 1 ):
    if binary:
        flags = 'wb'
    else:
        flags = 'w'
    try:
        with open( fName, flags ) as f: 
            pickle.dump( obj, f )
        print( "Saved" , fName , ', Exists?:', os.path.exists(fName), 
               ", Is file?:", os.path.isfile(fName) )
    except Exception as err:
        print( "Failed to save" , fName , '!\n' , err )

        
def get_pkl( fName, binary = 1 ):
    if binary:
        flags = 'rb'
    else:
        flags = 'r'
    try:
        with open( fName, flags ) as f: 
            f.seek(0)
            obj = pickle.load( f )
        return obj
    except Exception as err:
        print( "Failed to load" , fName , '!\n' , err )
        return None
    

########## Plot figure size ########################################################################
def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
