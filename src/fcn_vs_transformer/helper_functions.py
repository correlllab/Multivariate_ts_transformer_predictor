import numpy as np
import matplotlib.pyplot as plt

# Some helper functions
def graph_episode_output( res, index, ground_truth, out_decision, net, ts_ms = 20, save_fig=False ):
    """ Graph the result of `simulate_episode_input` """
    N = res.shape[0]
    L = res.transpose()
    X = np.arange(0, N*ts_ms, ts_ms) 
    plt.figure()
    # print( X.shape, L.shape )
    plt.stackplot(X, *L, labels=("Pr(Pass)", "Pr(Fail)"), baseline='zero')
    plt.axhline(y = 0.9, color = 'black', linestyle = '--')
    plt.axhline(y = 0.1, color = 'black', linestyle = '--')
    plt.legend()
    plt.axis('tight')
    plt.title(f'Probabilities for {net}: episode {index}\nGT = {ground_truth} | Decision = {out_decision}')
    
    if save_fig:
        plt.savefig(f'./imgs/{net}/{net}_prob_{index}.png')
        plt.clf()
    else:
        plt.show()
    
    
def scan_output_for_decision( output, trueLabel, threshold = 0.95 ):
    for i, row in enumerate( output ):
        if np.amax( row ) >= threshold:
            if np.dot( row, trueLabel ) >= threshold:
                ans = 'T'
            else:
                ans = 'F'
            if row[0] > row[1]:
                ans += 'P'
            else:
                ans += 'N'
            return ans, i
    return 'NC', output.shape[0]

def plot_FT( data_np, figsize=(18, 14), dpi=80, suppressT = 0 ):
    """ Interpret `data_np` as force-torque """
    x  = data_np[:,0]
    fx = data_np[:,1]
    fy = data_np[:,2]
    fz = data_np[:,3]
    tx = data_np[:,4]
    ty = data_np[:,5]
    tz = data_np[:,6]
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot( x , fx , label='Fx' )
    plt.plot( x , fy , label='Fy' )
    plt.plot( x , fz , label='Fz' )
    plt.xlabel( 'Time in millisecs' )
    plt.ylabel( 'Forces' )
    plt.title( 'Forces vs time ' )
    plt.legend()
    plt.show()

    if not suppressT:
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot( x , tx , label='Tx' )
        plt.plot( x , ty , label='Ty' )
        plt.plot( x , tz , label='Tz' )
        plt.xlabel( 'Time in millisecs' )
        plt.ylabel( 'Torques' )
        plt.title( 'Torques vs time ' )
        plt.legend()
        plt.show()