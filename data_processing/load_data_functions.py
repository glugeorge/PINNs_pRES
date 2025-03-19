import pandas as pd
import numpy as np

def load_elmer_results(csv,left_x,right_x):
    df = pd.read_csv(f'{csv}')
    keys = ['stress 1', 'stress 2', 'stress 4', 'strainrate 1', 'strainrate 2', 'strainrate 4', 'Points:0', 'Points:1','depth', 'height','velocity:0', 'velocity:1', 'pressure']
    new_indexes = ['txx', 'tzz', 'txz', 'exx', 'ezz', 'exz', 'x', 'z', 'd','h', 'u', 'w','p']
    df_cropped = df[keys]
    df_cropped.columns = new_indexes
    df_filtered = df_cropped[df_cropped['x'] >= left_x] 
    df_filtered = df_filtered[df_filtered['x']<= right_x]

    df_filtered['e_eff'] = np.sqrt(0.5*df_filtered['exx']**2 + 0.5*df_filtered['ezz']**2 + df_filtered['exz']**2)
    df_filtered['s_eff'] = np.sqrt(0.5*df_filtered['txx']**2 + 0.5*df_filtered['tzz']**2 + df_filtered['txz']**2)
    df_filtered['mu'] = 0.5*df_filtered['s_eff']/df_filtered['e_eff'] #[MPa yr]
    df_filtered['mu_xx'] = 0.5*df_filtered['txx']/df_filtered['exx'] #[MPa yr]
    df_filtered['mu_xz'] = 0.5*df_filtered['txz']/df_filtered['exz'] #[MPa yr]
    df_filtered['mu_zz'] = 0.5*df_filtered['tzz']/df_filtered['ezz'] #[MPa yr]
    
    df_filtered['rho'] = 910 # this was the constant density used in my elmer simulation
    
    df_surface = df_filtered[df_filtered['d']==0]
    df_divide = df_filtered[df_filtered['x']==0]
    df_bed = df_filtered[df_filtered['z']==0]
    left_x = df_filtered['x'].min()
    right_x = df_filtered['x'].max()
    
    df_flanks = df_filtered[(df_filtered['x'] == left_x) | (df_filtered['x'] == right_x)]
    return df_filtered, df_surface, df_divide, df_bed, df_flanks