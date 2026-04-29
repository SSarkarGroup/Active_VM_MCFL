#!/usr/bin/python
import os
import sys
from time import time
import numpy as np
from force import get_forces, get_forces_polys, move_vertices_vectorized
from transition import T1_transition_vectorized_new, division_transition, node_switch_optimized_new
from geometry import *
from motility import update_theta
from parameters import get_parameters
from parser import *
from periodic_hex_lattice_diff import generate_periodic_hex_lattice
from nematic import get_shared_edge2
from mechanochemical_stable import init_chem, update_chem, update_cyclin


def parse_args(argv):
    if len(argv) < 6:
        raise SystemExit("Usage: MD_diff.py <r> <Ahat> <cth> <seed> <output_dir>")

    r = int(argv[1])
    areahat = float(argv[2])
    cth = int(argv[3])
    seed = int(argv[4])
    filepath = argv[5]

    return r, areahat, cth, seed, filepath


def init_lattice(run_dir, lx, ly, numx, numy):
    generate_periodic_hex_lattice(lx, ly, numx, numy, run_dir)


def molecular_dynamics(
    vertices,
    edges,
    polys,
    parameters,
    total_time,
    folder,
    datadir,
    edge_dict,
    div_polys,
    div_times,
    rate_0,
    prob_0,
):
    delta_t = parameters["delta_t"]
    lx = parameters["lx"]
    ly = parameters["ly"]
    domain = np.array([lx, ly])

    t = 0.0
    count = 0

    while t < total_time:
        if prob_0 >= 1:
            prob_0 = 0

        if t > 100:
            polys, edges, edge_dict, vertices, div_polys, div_times, prob_0, n_div = division_transition(polys, edges, edge_dict, vertices, domain, parameters, div_polys, div_times, t, rate_0, prob_0)

            if n_div > 0:
                edges = np.unique(edges, axis=0)
                edges = add_reverse_edges(edges)
                edge_dict = get_shared_edge2(edges, polys)

        vertices, polys, edges, edge_dict, parameters, _ = T1_transition_vectorized_new(vertices, polys, edges, edge_dict, parameters)

        if count % 10 == 0:
            polys, edges, edge_dict, sw, new_edges, removed_edges = node_switch_optimized_new(vertices, polys, edges, edge_dict, domain)
            if sw:
                edge_dict = get_shared_edge2(edges, polys)

        forces, mot_forces, elas_forces = get_forces(vertices, polys, edges, edge_dict, parameters)

        polys = update_chem(polys, vertices, parameters, domain)
        polys = update_theta(polys, vertices, parameters, domain)
        polys = update_cyclin(polys, parameters, delta_t)

        vertices = move_vertices_vectorized(vertices, forces, parameters)

        if np.isnan(vertices).any():
            print("Nan in vertices")

        if count % 100 == 0:
            print('%d/%d, %0.2f %%,' % (int(t), int(total_time), 100*(t/total_time)), np.sum(forces ** 2) ** 0.5)
            datafile = os.path.join(datadir, "data_%.2f.npz" % (t))
            np.savez(datafile, vertices=vertices, polys=polys, edges=edges, L=domain, forces=forces)

        count += 1
        t += delta_t


def write_parameters_file(path, parameters, values):
    lines = [
        "lx: %.6f\n" % (values["lx"]),
        "ly: %.6f\n" % (values["ly"]),
        "ka: %.6f\n" % (values["ka"]),
        "gamma: %.6f\n" % (values["gamma"]),
        "lmin: %.6f\n" % (values["lmin"]),
        "delta_t: %.6f\n" % (values["delta_t"]),
        "eta: %.6f\n" % (values["eta"]),
        "xi: %.6f\n" % (values["xi"]),
        "zeta: %.6f\n" % (values["zeta"]),
        "tau: %.6f\n" % (values["tau"]),
        "strthr: %.6f\n" % (values["strthr"]),
        "D: %.6f\n" % (values["D"]),
        "rate_0: %.6f\n" % (values["rate_0"]),
        "P0: %.6f\n" % (values["P0"]),
        "q0: %.6f\n" % (values["q0"]),
        "alpha: %.6f\n" % (values["alpha"]),
        "beta: %.6f\n" % (values["beta"]),
        "A0hat: %.6f\n" % (parameters["A0hat"]),
        "Ahat: %.6f\n" % (parameters["Ahat"]),
        "tauD: %.6f\n" % (values["tauD"]),
        "tauA: %.6f\n" % (values["tauA"]),
        "tauP: %.6f\n" % (values["tauP"]),
        "prefactor: %.6f\n" % (values["prefactor"]),
        "seed: %d\n" % (values["seed"]),
    ]

    with open(path, "w") as file_handle:
        file_handle.writelines(lines)


def main():
    print("sys.argv:", sys.argv)

    r, areahat, cth, seed, filepath = parse_args(sys.argv)
    np.random.seed(seed)

    name = "r_%d_cth_%d_areahat_%0.2f_seed_%d" % (r, cth, areahat, seed)
    run_dir = os.path.join(filepath, name)
    folder = os.path.join(run_dir, "figure")
    datadir = os.path.join(run_dir, "raw")

    print("dir:", run_dir)
    os.makedirs(run_dir, exist_ok=True)
    print("filepath:", filepath)

    lx = 20
    ly = 20
    numx = 18
    numy = 18
    init_lattice(run_dir, lx, ly, numx, numy)

    os.makedirs(folder, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    vertex_file = os.path.join(run_dir, "vertices.txt")
    edge_file = os.path.join(run_dir, "edges.txt")
    poly_file = os.path.join(run_dir, "cell_indices.txt")
    domain = np.loadtxt(os.path.join(run_dir, "L.txt"))
    lx = domain[0]
    ly = domain[1]
    area0 = np.loadtxt(os.path.join(run_dir, "area.txt"))

    q0 = 3.9
    zeta = 0.1
    P0 = q0 * np.sqrt(area0)
    q = P0

    print("lx:", lx, "ly:", ly, "P0:", P0, "A0:", area0, "q0:", q0, "q:", q)

    ka = 1.2
    A0 = area0
    gamma = 0.38
    lmin = 0.01
    delta_t = 0.01
    eta = 0
    xi = 0.125
    tau = 0.01
    strthr = 0.1
    D = 0.5

    print("ka:", ka, "gamma:", gamma, "A0:", A0, "P0:", P0)

    div_polys = []
    div_times = []
    rate_0 = r * 1e-5
    prob_0 = 0

    alpha = 2.25
    beta = 2.88
    tauD = 120
    tauA = 120
    tauP = 120
    prefactor = 2

    total_time = 1600.0

    vertices = read_vertices(vertex_file)
    m0, n0 = vertices.shape
    vertices = vertices + 0.55 * np.random.rand(m0, n0)

    edges = read_edges(edge_file)
    edges = np.unique(edges, axis=0)

    poly_indices = read_poly_indices(poly_file)
    polys = build_polygons(poly_indices, A0, P0, xi)

    print("Number of polygons:", len(polys))

    edge_dict = get_shared_edge2(edges, polys)

    for poly in polys:
        poly.cellType = 0

    parameters = get_parameters(lx, ly, ka, gamma, eta, xi, lmin, delta_t, zeta, tau, q)

    parameters["D"] = D
    parameters["q0"] = q0

    parameters["A0hat"] = area0
    parameters["Ahat"] = areahat
    parameters["cth"] = cth

    parameters["alpha"] = alpha
    parameters["beta"] = beta
    parameters["tauD"] = tauD
    parameters["tauA"] = tauA
    parameters["tauP"] = tauP
    parameters["pf"] = prefactor

    values = {
        "lx": lx,
        "ly": ly,
        "ka": ka,
        "gamma": gamma,
        "lmin": lmin,
        "delta_t": delta_t,
        "eta": eta,
        "xi": xi,
        "zeta": zeta,
        "tau": tau,
        "strthr": strthr,
        "D": D,
        "rate_0": rate_0,
        "P0": P0,
        "q0": q0,
        "alpha": alpha,
        "beta": beta,
        "tauD": tauD,
        "tauA": tauA,
        "tauP": tauP,
        "prefactor": prefactor,
        "seed": seed,
    }

    paramfile = os.path.join(run_dir, "parameters.txt")
    write_parameters_file(paramfile, parameters, values)

    time1 = time()
    ncell1 = len(polys)

    init_chem(polys)
    molecular_dynamics(vertices, edges, polys, parameters, total_time, folder, datadir, edge_dict, div_polys, div_times, rate_0, prob_0)

    time2 = time()
    ncell2 = len(polys)

    np.savetxt(os.path.join(run_dir, "time.txt"), [time2 - time1])
    np.savetxt(os.path.join(run_dir, "ncell.txt"), [ncell1, ncell2])


if __name__ == "__main__":
    main()
