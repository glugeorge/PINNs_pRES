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
    # get scales
    x0 = scales['x_range']
    z0 = scales['z_range']
    w0 = scales['w_range']
    mu0 = scales['mu_range']

    def grad1stOrder(net, x):
        grad, sol = vectgrad(net, x)
        # order should u w rho p mu
        rho = sol[:, 2:3]
        mu = sol[:,4:5]

        u_x = grad[:, 0:1]
        u_z = grad[:, 1:2]
        w_x = grad[:, 2:3]
        w_z = grad[:, 3:4]
        # rho_x, rho_z for 4:5, 5:6
        p_x = grad[:, 6:7]
        p_z = grad[:, 7:8]

        term1_1 = mu*u_x
        term12_21 = mu*(u_z/z0 + w_x/x0 )
        term1_3 = p_x

        term2_2 = mu*w_z
        term2_3 = p_z + rho

        return jnp.hstack([term1_1,term12_21,term1_3,term2_2,term2_3])

    func_g = lambda x: grad1stOrder(net, x)
    grad_term, term = vectgrad(func_g, x)
    
    e1term1 = 2*mu0*w0*grad_term[:, 0:1]/(910*9.81*z0*x0) # (term1_1,x)
    e1term2 = mu0*w0*x0*grad_term[:,3:4]/(910*9.81*z0**2) # (term12_21,z)
    e1term3 = term[:,2:3]
    e2term1 = mu0*w0*grad_term[:,2:3]/(910*9.81*x0) # (term12_21,x)
    e2term2 = 2*mu0*w0*grad_term[:,7:8]/(910*9.81*z0**2)
    e2term3 = term[:,4:5]

    e1 = e1term1 + e1term2 - e1term3
    e2 = e2term1 + e2term2 - e2term3
    f_eqn = jnp.hstack([e1, e2])
    terms = jnp.hstack([e1term1, e1term2, e1term3, e2term1, e2term2, e2term3])
    return f_eqn, terms