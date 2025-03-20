import sys
import os
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gradients import vgmat, vectgrad

'''
Across all data 0:1=u, 1:2=w, 2:3 =rho, 3:4= p, 4:5=mu
Gradients: 0:1 = u_x, 1:2 = u_z, 2:3 = w_x, 3:4 = w_z, 4:5 = rho_x, 5:6 = rho_z, 6:7 = p_x, 7:8 = p_z, 8:9 = mu_x, 9:10 = mu_z
'''

def gov_eqn(net, x, scales):
    grad, sol = vectgrad(net,x)

    # Load in relevant variables
    u_x = grad[:,0:1]
    w_z = grad[:,3:4]
    rho_z = grad[:,5:6]
    w = sol[:,1:2]
    rho = sol[:,2:3]

    # Load in relevant scales for equations
    z_range = scales['z_range']
    x_range = scales['x_range']
    w_mean = scales['w_mean']
    w_range = scales['w_range']

    eterm1 = u_x * z_range/x_range
    eterm2 = (w+w_mean/w_range)*rho_z + rho*w_z
    e1 = eterm1 + eterm2
    terms = jnp.hstack([eterm1,eterm2])
    return e1,terms