import numpy as np
from geometry import *
from numba import njit
from numba.typed import List


def get_forces(vertices, polys, edges, edgeDict, parameters):
    _ = edges, edgeDict
    lx = parameters["lx"]
    ly = parameters["ly"]
    L = np.array([lx, ly])
    ka = parameters["ka"]
    gamma = parameters["gamma"]
    xi = parameters["xi"]

    f1 = F_elasticity_vectorized(vertices, polys, ka, L)
    f3 = F_contraction_vectorized(vertices, polys, gamma, L)
    f4 = F_motility(vertices, polys, xi, L)

    return -(f1 + f3 - f4), f4, -(f1 + f3)


def get_forces_polys(vertices, polys, forces):
    poly_force = np.zeros((len(polys), 2))
    for i, poly in enumerate(polys):
        indices = poly.indices
        for j in indices:
            poly_force[i, :] += forces[j]

    return poly_force


def move_vertices_vectorized(vertices, forces, parameters):
    delta_t = parameters["delta_t"]
    lx = parameters["lx"]
    ly = parameters["ly"]

    vertices = vertices + delta_t * forces
    vertices[:, 0] = np.mod(vertices[:, 0], lx)
    vertices[:, 1] = np.mod(vertices[:, 1], ly)

    return vertices


@njit(cache=True)
def F_elasticity_fast(vertices, poly_indices_list, poly_A0_list, poly_areas, ka, L):
    forces = np.zeros((len(vertices), 2))
    n_polys = len(poly_indices_list)

    for poly_idx in range(n_polys):
        poly_indices = poly_indices_list[poly_idx]
        coeff = ka * (poly_A0_list[poly_idx] - poly_areas[poly_idx])
        n_verts = len(poly_indices)

        for j in range(n_verts):
            i = poly_indices[j]
            cw_pos = (j + 1) % n_verts
            v0 = vertices[i]
            v_cw = vertices[poly_indices[cw_pos]]
            vc = v0 + periodic_diff(v_cw, v0, L)

            ccw_pos = (j - 1) % n_verts
            v_ccw = vertices[poly_indices[ccw_pos]]
            vcc = v0 + periodic_diff(v_ccw, v0, L)

            diff = vc - vcc
            f_x = -0.5 * diff[1]
            f_y = 0.5 * diff[0]

            forces[i, 0] += coeff * f_x
            forces[i, 1] += coeff * f_y

    return forces


def F_elasticity_vectorized(vertices, polys, ka, L):
    poly_indices_list = List()
    poly_A0_list = np.empty(len(polys))
    poly_areas = np.empty(len(polys))

    for idx, poly in enumerate(polys):
        poly_indices_list.append(np.asarray(poly.indices, dtype=np.int64))
        poly_A0_list[idx] = poly.A0
        poly_areas[idx] = poly.get_area(vertices, L)

    return F_elasticity_fast(vertices, poly_indices_list, poly_A0_list, poly_areas, ka, L)


@njit(cache=True)
def F_contraction_fast(vertices, poly_indices_list, poly_gamma_list, perimeters, P0_values, L):
    forces = np.zeros((len(vertices), 2))
    n_polys = len(poly_indices_list)

    for poly_idx in range(n_polys):
        poly_indices = poly_indices_list[poly_idx]
        coeff = poly_gamma_list[poly_idx] * (perimeters[poly_idx] - P0_values[poly_idx])
        n_verts = len(poly_indices)

        for j in range(n_verts):
            i = poly_indices[j]
            v0 = vertices[i]

            cw_pos = (j + 1) % n_verts
            v_cw = vertices[poly_indices[cw_pos]]
            vc = v0 + periodic_diff(v_cw, v0, L)

            ccw_pos = (j - 1) % n_verts
            v_ccw = vertices[poly_indices[ccw_pos]]
            vcc = v0 + periodic_diff(v_ccw, v0, L)

            uvc = unit_vector(v0, vc)
            uvcc = unit_vector(vcc, v0)

            forces[i, 0] += coeff * (uvc[0] - uvcc[0])
            forces[i, 1] += coeff * (uvc[1] - uvcc[1])

    return forces


def F_contraction_vectorized(vertices, polys, gamma, L):
    poly_indices_list = List()
    poly_gamma_list = np.empty(len(polys))
    perimeters = np.empty(len(polys))
    P0_values = np.empty(len(polys))

    for idx, poly in enumerate(polys):
        poly_indices_list.append(np.asarray(poly.indices, dtype=np.int64))
        poly_gamma_list[idx] = poly.Gam
        perimeters[idx] = poly.get_perim(vertices, L)
        P0_values[idx] = poly.P0

    return F_contraction_fast(vertices, poly_indices_list, poly_gamma_list, perimeters, P0_values, L)


@njit(cache=True)
def F_motility_fast(n_vertices, poly_indices_list, polarity_array):
    forces = np.zeros((n_vertices, 2))
    n_polys = len(poly_indices_list)

    for poly_idx in range(n_polys):
        poly_indices = poly_indices_list[poly_idx]
        px = polarity_array[poly_idx, 0]
        py = polarity_array[poly_idx, 1]

        for j in range(len(poly_indices)):
            i = poly_indices[j]
            forces[i, 0] += px
            forces[i, 1] += py

    return forces


def F_motility(vertices, polys, xi, L):
    _ = xi, L
    poly_indices_list = List()
    polarity_array = np.empty((len(polys), 2))

    for idx, poly in enumerate(polys):
        poly_indices_list.append(np.asarray(poly.indices, dtype=np.int64))
        polarity_array[idx, :] = poly.p

    return F_motility_fast(len(vertices), poly_indices_list, polarity_array)
