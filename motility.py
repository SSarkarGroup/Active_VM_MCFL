import numpy as np
from numba import njit


@njit(cache=True)
def update_theta_fast(p_array, delta_t, D):
    n_polys = p_array.shape[0]
    new_theta = np.empty(n_polys)
    new_p = np.empty((n_polys, 2))
    sqrt_2D = np.sqrt(2 * D)
    
    for i in range(n_polys):
        px = p_array[i, 0]
        py = p_array[i, 1]
        
        theta = np.arctan2(py, px)

        dthetadt = sqrt_2D * np.random.randn()
        
        new_theta[i] = theta + dthetadt * delta_t
        
        new_p[i, 0] = np.cos(new_theta[i])
        new_p[i, 1] = np.sin(new_theta[i])
    
    return new_theta, new_p


def update_theta(polys, vertices, parameters, L):

    delta_t = parameters['delta_t']
    D = parameters['D']
    xi = parameters['xi']
    Ahat = parameters['Ahat']
    
    n_polys = len(polys)
    p_array = np.empty((n_polys, 2))
    for i, poly in enumerate(polys):
        p_array[i, :] = poly.p
    
    new_theta, new_p = update_theta_fast(p_array, delta_t, D)
    
    for i, poly in enumerate(polys):
        A = poly.get_area(vertices, L)
        poly.theta = new_theta[i]
        p_b = 1 / (1 + (A/Ahat)**2)
        v = 2 * xi * (1-p_b)
        poly.p = v * new_p[i, :]
    
    return polys