#!/usr/bin/python
import numpy as np
from geometry import *
from numba import njit
from numba import int32, float64    # import the types
from numba.experimental import jitclass


@njit(cache=True)
def get_poly_vertices_fast(indices, vertices, L):
	n = len(indices)
	poly_vertices = np.empty((n, 2), dtype=np.float64)
	
	# Get first vertex
	first_idx = indices[0]
	v_last = np.array([vertices[first_idx, 0], vertices[first_idx, 1]])
	poly_vertices[0, 0] = v_last[0]
	poly_vertices[0, 1] = v_last[1]
	
	# Process remaining vertices
	for j in range(1, n):
		i = indices[j]
		v = np.array([vertices[i, 0], vertices[i, 1]])
		v_next = v_last + periodic_diff(v, v_last, L)
		poly_vertices[j, 0] = v_next[0]
		poly_vertices[j, 1] = v_next[1]
		v_last = v_next
	
	return poly_vertices


class Polygon:

	def __init__(self, id, indices, A0, P0, xi):
		self.id = id
		self.indices = indices
		self.A0 = A0
		self.theta = np.random.uniform(0,2*np.pi)
		self.p = xi*np.array([np.cos(self.theta), np.sin(self.theta)])
		self.nd = np.zeros(([2,1]))
		self.center = np.zeros(([2,1]))
		self.P0 = P0
		self.cellType = 0
		self.E = 2.1
		self.M = 0.9
		self.D = 0.1
		self.a = 1
		self.d0 = 0.1
		self.E0 = 2.1
		self.Gam = 0.38
		self.c = 0.5 
		self.alpha = 2.25
		self.beta = 2.88

		self.cellid = id

	# return list of vertices
	# with periodic boundaries 
	def get_poly_vertices(self, vertices, L):
		# Convert indices to numpy array if needed and call fast Numba function
		indices_arr = np.asarray(self.indices, dtype=np.int64)
		return get_poly_vertices_fast(indices_arr, vertices, L)

	def get_area(self, vertices, L):
		poly_vertices = self.get_poly_vertices(vertices, L)
		a = area_periodic(poly_vertices, L)
		return a 
	
	def get_perim(self, vertices, L):
		poly_vertices = self.get_poly_vertices(vertices, L)
		p = perimeter(poly_vertices)
		return p

	def get_center(self, vertices, L):
		x,y = center(self.get_poly_vertices(vertices, L))
		self.center = np.array(([x,y]))
		return x,y
	
	def set_indices(self, indices):
		self.indices = indices

	def get_nematic_director(self,vertices,L):
		# xC,yC = self.get_center(vertices,L) 
		poly_vertices = self.get_poly_vertices(vertices, L)
		nd,eval  = nematic_director(poly_vertices)
		self.nd = nd.flatten()
		return nd,eval 
	
	def get_asphericity(self,vertices,L):
		poly_vertices = self.get_poly_vertices(vertices, L)
		nd,eval  = nematic_director(poly_vertices)
		asph = np.abs(eval[0] - eval[1])/np.abs(eval[0] + eval[1])
		return asph 
	
	def get_hpressure(self, vertices, L):
		area = self.get_area(vertices, L)
		KA = 1
		A0 = self.A0
		hpressure = -KA * (area - A0)
		return hpressure

	def get_polarity(self, vertices, t_signs, L):
		id = self.id
		poly_vertices = self.get_poly_vertices(vertices, L)
		nd,eval  = nematic_director(poly_vertices)
		polarity = nd * t_signs[id]
		return polarity
