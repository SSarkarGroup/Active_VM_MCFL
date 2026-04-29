#!/usr/bin/python
import numpy as np
from Polygon import Polygon
from geometry import rand_angle


def read_poly_indices(file):
	indices = []
	f = open(file)
	for line in f:
		poly_indices = []
		linesplit = line.strip().split("\t")
		for i in linesplit:
			poly_indices.append(int(i))
		indices.append(poly_indices)
		
	f.close()
	return indices


def build_polygons(cell_indices, A0,P0, xi):
	polys = []
	for i,indices in enumerate(cell_indices):
		rand_angle()
		poly = Polygon(i, indices, A0, P0, xi)
		poly.P0 = P0
		polys.append(poly)
	return polys


def read_vertices(file):
	vertices = np.loadtxt(file)
	return vertices


def read_edges(file):
	edges = np.loadtxt(file).astype(int)
	return edges


def check_presence(array, element):
    for pair in array:
        if np.array_equal(pair, element):
            return True
    return False


def find_indices_without_reverse(edges):
	indices = []
	for i, edge in enumerate(edges):
		if not check_presence(edges, [edge[1], edge[0]]):
			indices.append(i)
	return indices


def add_reverse_edges(edges):
	indices_without_reverse = find_indices_without_reverse(edges)
	reverse_edges = []

	for idx in indices_without_reverse:
		edge = edges[idx]
		reverse_edge = [edge[1], edge[0]]
		reverse_edges.append(reverse_edge)

	if len(reverse_edges) > 0:
		edges = np.vstack([edges, reverse_edges])
	return edges




