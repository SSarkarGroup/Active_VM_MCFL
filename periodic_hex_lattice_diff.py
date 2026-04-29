import os
import numpy as np
from grid import generate_periodic_grid
from trace_periodic import trace_periodic_vertices


def generate_periodic_hex_lattice(nx, ny, numx, numy, filepath):
	nx = float(nx)
	ny = float(ny)
	numx = float(numx)
	numy = float(numy)

	if numx % 2 != 0:
		print("Number of hexagons on x axis should be a multiple of 2")
		numx += 1

	frac = numx / nx

	s = np.sqrt(2) / np.sqrt(3 * np.sqrt(3))
	print("Side length of hexagon: ", s)

	s = s / frac

	A = (np.sqrt(3) * 3 / 2.0) * s**2

	w = 2.0 * s

	h = (np.sqrt(3) / 2.0) * w

	np.savetxt(os.path.join(filepath, "nx_ny.txt"), [nx, ny], fmt="%.18e")

	np.savetxt(os.path.join(filepath, "numx_numy.txt"), [numx, numy], fmt="%.18e")

	np.savetxt(os.path.join(filepath, "area.txt"), [A], fmt="%.18e")

	xx, yy, L = generate_periodic_grid(int(numx), int(numy), w, h)
	print("filepath: ", filepath)
	np.savetxt(os.path.join(filepath, "L.txt"), L)

	vertices, indices = trace_periodic_vertices(int(numx), int(numy), xx, yy, w, h, L)

	np.savetxt(os.path.join(filepath, "vertices.txt"), vertices)

	with open(os.path.join(filepath, "edges.txt"), "w+") as file_handle:
		for index in indices:
			for i in range(0, 5):
				i1 = int(index[i])
				i2 = int(index[i + 1])
				file_handle.write("%d \t %d\n" % (i1, i2))
			i1 = int(index[-1])
			i2 = int(index[0])
			file_handle.write("%d \t %d\n" % (i1, i2))

	np.savetxt(os.path.join(filepath, "cell_indices.txt"), indices, delimiter="\t", fmt="%d")


