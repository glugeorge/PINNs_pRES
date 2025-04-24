import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp
from jax import random
import pandas as pd


# Sampling from normalized data
def sample(df_norm,n_samples,df_bcs):
    # df_norm is the dataframe of interest
    # key is the rng key that we set for reproducibility 
    # n_samples is a list including number of samples, followed by number of colocation points
    def dataf(key):
        keys = random.split(key, 2) 
        X_star = df_norm[['x','z']].values
        U_star = df_norm[['u','w','rho','p','mu']].values
        n_data = X_star.shape[0]    
        idx_smp = random.choice(keys[0], jnp.arange(n_data), [n_samples[0]])
        X_smp = X_star[idx_smp,:]
        U_smp = U_star[idx_smp,:]
    
        idx_col = random.choice(keys[1], jnp.arange(n_data), [n_samples[1]])
        X_col = X_star[idx_col,:]
        U_col = U_star[idx_col,:]
        data = dict(smp=[X_smp,U_smp],col=[X_col,U_col])

        # add in bcs - currently, include all data
        for dict_key, df_bc in zip(['surf','div','bed','flanks'],df_bcs):
            X_bc = df_bc[['x','z']].values
            U_bc = df_bc[['u','w','rho','p','mu']].values
            data[dict_key] = [X_bc,U_bc]
        
        return data
    return dataf

# Normalizing input data
def normalize(df_filtered, df_surface, df_divide, df_bed, df_flanks):
    # keys that will always be there
    x = df_filtered['x'].values
    z = df_filtered['z'].values
    w = df_filtered['w'].values
    
    rho_bed = df_bed['rho'].values.mean()
    mu_flanks = df_flanks['mu'].values
    
    # assign scales - use a dict
    means_and_scales = {}
    means_and_scales['x_mean'] = jnp.mean(x)
    means_and_scales['x_range'] = (x.max() - x.min()) / 2
    means_and_scales['z_mean'] = jnp.mean(z)
    means_and_scales['z_range'] = (z.max() - z.min()) / 2
    means_and_scales['w_mean'] = jnp.mean(w)
    means_and_scales['w_range'] = jnp.std(w) * 2
    # the divide assumes that u will vary around 0
    means_and_scales['u_mean'] = 0
    means_and_scales['u_range'] = jnp.std(w) * 2 # same as w_range
    # let's say the ice scaling is that of glacial ice (elmer uses 910)
    # let's also say the pressure scale is dependent on ice thickness
    means_and_scales['rho_range'] = rho_bed
    means_and_scales['rho_mean'] = 0
    means_and_scales['p_range'] = means_and_scales['rho_range']*9.81*means_and_scales['z_range'] # this also assumes not centered at 0 
    means_and_scales['p_mean'] = 0
    means_and_scales['mu_range'] = jnp.exp(jnp.nanmean(jnp.log(jnp.abs(mu_flanks))))
    means_and_scales['mu_mean'] = 0

    #norm_list
    df_full_norm,df_surf_norm,df_div_norm,df_bed_norm,df_flanks_norm = pd.DataFrame([]),pd.DataFrame([]),pd.DataFrame([]),pd.DataFrame([]),pd.DataFrame([])
    
    for key in ['x','z','u','w','rho','p','mu']:
        for df,df_norm in zip([df_filtered,df_surface, df_divide, df_bed, df_flanks],[df_full_norm,df_surf_norm,df_div_norm,df_bed_norm,df_flanks_norm]):
            df_norm[key] = (df[key] - means_and_scales[f'{key}_mean'])/means_and_scales[f'{key}_range']
    return df_full_norm,[df_surf_norm,df_div_norm,df_bed_norm,df_flanks_norm],means_and_scales

def gradient_z(arr, z_values):
    """
    Computes the gradient in the z-direction for a 2D array with NaNs.
    
    Parameters:
        arr (numpy.ndarray): m x n array with NaNs.
        z_values (numpy.ndarray): m x n array giving the z-coordinates for each element.

    Returns:
        numpy.ndarray: m x n array with gradients, preserving NaN structure.
    """
    m, n = arr.shape
    grad_arr = np.full_like(arr, np.nan)  # Initialize gradient array with NaNs

    for j in range(n):  # Iterate over columns
        col = arr[:, j]
        z_col = z_values[:, j]

        # Get indices of non-NaN values
        valid_idx = np.where(~np.isnan(col))[0]
        
        if len(valid_idx) > 1:
            valid_z = z_col[valid_idx]  # Get corresponding z-coordinates
            valid_col = col[valid_idx]  # Get corresponding values
            
            # Compute gradient w.r.t. z spacing
            grad_valid = np.gradient(valid_col, valid_z)
            
            # Place computed gradient values back into original structure
            grad_arr[valid_idx, j] = grad_valid

    return grad_arr

def interpolate_columns(arr, z_values, z_interp):
    """
    Interpolates each column onto a uniform z grid.

    Parameters:
        arr (numpy.ndarray): m x n array with NaNs.
        z_values (numpy.ndarray): m x n array of z-coordinates.
        z_interp (numpy.ndarray): 1D array of uniform z-coordinates.

    Returns:
        numpy.ndarray: Interpolated array with shape (len(z_interp), n).
    """
    m, n = arr.shape
    interp_arr = np.full((len(z_interp), n), np.nan)

    for j in range(n):  # Iterate over columns
        col = arr[:, j]
        z_col = z_values[:, j]

        # Get valid values
        valid_idx = ~np.isnan(col)
        if np.sum(valid_idx) < 2:
            continue  # Skip columns with <2 points

        # Interpolation function
        f = interp1d(z_col[valid_idx], col[valid_idx], kind='linear', bounds_error=False, fill_value=np.nan)
        interp_arr[:, j] = f(z_interp)  # Interpolate column-wise

    return interp_arr

def gradient_x(arr, x_values, z_values, num_z_interp=200):
    """
    Computes the gradient in the x-direction for a sparse 2D array.

    Parameters:
        arr (numpy.ndarray): m x n array with NaNs.
        x_values (numpy.ndarray): m x n array of x-coordinates.
        z_values (numpy.ndarray): m x n array of z-coordinates.
        num_z_interp (int): Number of points for interpolation in z.

    Returns:
        numpy.ndarray: m x n array with gradient values.
    """
    # Define uniform z grid for interpolation
    z_min, z_max = np.nanmin(z_values), np.nanmax(z_values)
    z_interp = np.linspace(z_min, z_max, num_z_interp)

    # Interpolate in the z direction
    interp_arr = interpolate_columns(arr, z_values, z_interp)

    # Interpolate x_values to match the new grid shape
    x_interp = np.tile(x_values[0, :], (num_z_interp, 1))

    # Compute gradient in x-direction
    grad_x_interp = np.full_like(interp_arr, np.nan)

    for i in range(num_z_interp):
        row = interp_arr[i, :]
        valid_idx = ~np.isnan(row)
        
        if np.sum(valid_idx) < 2:
            continue  # Skip rows with insufficient data

        grad_x_interp[i, valid_idx] = np.gradient(row[valid_idx], x_interp[i, valid_idx])

    # Map gradient values back to the original structure
    grad_x_original = np.full_like(arr, np.nan)

    for j in range(arr.shape[1]):  # Iterate over columns
        valid_idx = ~np.isnan(z_values[:, j])
        if np.sum(valid_idx) < 2:
            continue  # Skip if too few valid points

        f = interp1d(z_interp, grad_x_interp[:, j], kind='linear', bounds_error=False, fill_value=np.nan)
        grad_x_original[valid_idx, j] = f(z_values[valid_idx, j])  # Map back to original structure

    return grad_x_original
