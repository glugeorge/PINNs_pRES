import jax.numpy as jnp
from jax import vjp, vmap, lax

def vgmat(x, n_out, idx=None):
    '''
    :param n_out: number of output variables
    :param idx: indice (list) of the output variable to take the gradient
    '''
    if idx is None:
        idx = range(n_out)
    # obtain the number of index
    n_idx = len(idx)
    # obtain the number of input points
    n_pt = x.shape[0]
    # determine the shape of the gradient matrix
    mat_shape = [n_idx, n_pt, n_out]
    # create the zero matrix based on the shape
    mat = jnp.zeros(mat_shape)
    # choose the associated element in the matrix to 1
    for l, ii in zip(range(n_idx), idx):
        mat = mat.at[l, :, ii].set(1.)
    return mat


# vector gradient of the output with input
def vectgrad(func, x):
    # obtain the output and the gradient function
    sol, vjp_fn = vjp(func, x)
    # determine the mat grad
    mat = vgmat(x, sol.shape[1])
    # calculate the gradient of each output with respect to each input
    grad0 = vmap(vjp_fn, in_axes=0)(mat)[0]
    # calculate the total partial derivative of output with input
    n_pd = x.shape[1] * sol.shape[1]
    # reshape the derivative of output with input
    grad = grad0.transpose(1, 0, 2)
    grad_all = grad.reshape(x.shape[0], n_pd)
    return grad_all, sol