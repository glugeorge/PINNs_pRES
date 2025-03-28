from jax import random
import jax.numpy as jnp

# define the basic formation of neural network
def neural_net(params, x, scl, act_s=0):
    '''
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param sgn:  1 for even function and -1 for odd function
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    # choose the activation function
    actv = [jnp.tanh, jnp.sin][act_s]
    # normalize the input
    H = x  # input has been normalized
    # separate the first, hidden and last layers
    first, *hidden, last = params
    # calculate the first layers output with right scale
    H = actv(jnp.dot(H, first[0]) * scl + first[1])
    # calculate the middle layers output
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    # no activation function for last layer
    var = jnp.dot(H, last[0]) + last[1]
    return var
    
# initialize single network
def init_single_net(parent_key, layer_widths):
    params = []
    keys = random.split(parent_key, num=len(layer_widths) - 1)
    # create the weights and biases for the network
    for in_dim, out_dim, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = random.split(key)
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * 0]
        )
    return params

def init_pinns(parent_key, n_hl, n_unit): 
    # This should be general! Try to get solution/loss functions to adapt to this
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''

    # set the neural network shape for u, w
    layers1 = [2] + n_hl * [n_unit] + [2] 

    # set the neural network shape for p, mu
    layers2 = [2] + n_hl * [n_unit] + [2] 

    # generate the random key for each network
    keys = random.split(parent_key, 3)
    
    # generate weights and biases for density
    params_uw = init_single_net(keys[0], layers1)
    params_pmu = init_single_net(keys[1], layers2)

    params_rho = random.truncated_normal(keys[2], -2, 2, shape=(2,)) # 2 params rho_s and L

    return [params_uw, params_rho, params_pmu]

def create_solution(df_surf,scl=1, act_s=0):
    # this should also be a general function!
    # df_surf is nondimensional
    x_s = df_surf['x'].values
    z_s = df_surf['z'].values
    
    def f(params,x):
        x0, z0 = jnp.split(x, 2, axis=1)
        uw = neural_net(params[0], x, scl, act_s)
        pm_rho = params[1]
        H = jnp.interp(x0,x_s,z_s)  
        rho = 1 + (jnp.exp(pm_rho[0]) - 1) * jnp.exp((-H+z0)/jnp.exp(pm_rho[1]))
        pmu = neural_net(params[2], x, scl, act_s)
        sol = jnp.hstack([uw,rho,pmu])
        return sol
    return f