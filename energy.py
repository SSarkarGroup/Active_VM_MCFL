import numpy as np


def get_energy(vertices, polys, edges, parameters):
	# get necessary parameters 
	lx = parameters['lx']
	ly = parameters['ly']
	L = np.array([lx,ly])
	ka = parameters['ka']
	gamma = parameters['gamma']

	e1 = E_elasticity(vertices, polys, ka, L)
	e3 = E_contraction(vertices, polys, gamma, L)

	return (e1 + e3)


def E_elasticity(vertices, polys, ka, L):
	e = 0.
	for poly in polys:
		a = poly.get_area(vertices, L)
		A0 = poly.A0
		e += (ka / 2.) * (a - A0)**2
	return e


def E_contraction(vertices, polys, gamma, L):
	e = 0.
	for poly in polys:
		p = poly.get_perim(vertices, L)
		P0 = poly.P0
		e += ((gamma / 2.) * ((p-P0)**2))
	return e







