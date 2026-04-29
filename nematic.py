import numpy as np
from collections import defaultdict
from geometry import periodic_diff, unit_vector


def get_shared_edge(edges, polys):
    edgeDict = {}
    for edge in edges:
        sharedPoly = []
        for poly in polys:
            contains = np.sum(np.isin(poly.indices, edge)) == 2
            if contains:
                sharedPoly.append(poly.id)
        edgeDict[repr(edge)] = sharedPoly

    return edgeDict


def get_shared_edge2(edges, polys):
    vertex_pair_to_poly = defaultdict(list)

    for poly in polys:
        indices = poly.indices
        poly_id = poly.id
        num_vertices = len(indices)

        for i in range(num_vertices):
            v1, v2 = sorted((indices[i], indices[(i + 1) % num_vertices]))
            vertex_pair_to_poly[(v1, v2)].append(poly_id)

    edgeDict = {}
    for edge in edges:
        v1, v2 = sorted(edge)
        edge_key = repr(edge)
        edgeDict[edge_key] = vertex_pair_to_poly.get((v1, v2), [])
        if len(edgeDict[edge_key]) != 2:
            edgeDict[edge_key] = fix_edgeDict(edgeDict, edge, polys)

    return edgeDict


def check_consecutive_vertices(indices, v1, v2):
    num_vertices = len(indices)
    for i in range(num_vertices):
        if (indices[i] == v1 and indices[(i + 1) % num_vertices] == v2) or \
           (indices[i] == v2 and indices[(i + 1) % num_vertices] == v1):
            return True
    return False


def fix_edgeDict(edgeDict, edge, polys):
    try:
        if edge is None or len(edge) < 2:
            return []
        i1 = edge[0]
        i2 = edge[1]
        sharedPoly = []
        for poly in polys:
            contains = check_consecutive_vertices(poly.indices, i1, i2)
            if contains:
                sharedPoly.append(poly.id)
        edge_key = repr(edge)
        edgeDict[edge_key] = sharedPoly
        return sharedPoly
    except Exception:
        return []