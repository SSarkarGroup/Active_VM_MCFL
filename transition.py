#!/usr/bin/python
import copy
import numpy as np
from Polygon import Polygon as MyPolygon
from shapely.geometry import LineString
from scipy.spatial import KDTree
from numba import njit
from energy import get_energy
from geometry import (
    periodic_diff,
    unit_vector,
    isInside,
    isConvex,
    get_intersection_edge,
    is_point_in_polygon,
    move_point_to_nearest_edge,
)
from mechanochemical_stable import init_chem
from nematic import (
    get_shared_edge2,
    fix_edgeDict,
)

def get_poly_edges(poly, edges):
    poly_edges = []
    indices = poly.indices
    for edge in edges:
        if edge[0] in indices and edge[1] in indices:
            poly_edges.append(edge)
    return poly_edges

def get_6_indices(polys, i1, i2, poly_ids):
    p1_idx = list(polys[poly_ids[1]].indices)
    p3_idx = list(polys[poly_ids[3]].indices)
    
    pos1, pos2 = p1_idx.index(i1), p3_idx.index(i2)
    
    indices = [
        i1, i2,
        p1_idx[(pos1 - 1) % len(p1_idx)], p1_idx[(pos1 + 1) % len(p1_idx)],
        p3_idx[(pos2 - 1) % len(p3_idx)], p3_idx[(pos2 + 1) % len(p3_idx)]
    ]
    return [int(x) for x in indices]

def T1_0(polys, i1, i2, poly_ids, indices):
    i1, i2, i3, i4, i5, i6 = indices
    ps = [copy.copy(polys[pid]) for pid in poly_ids]
    for p in ps:
        p.indices = list(p.indices)
        
    edges = np.array([
        [i1, i2], [i1, i3], [i1, i4], [i2, i1], [i2, i5],
        [i2, i6], [i3, i1], [i4, i1], [i5, i2], [i6, i2]
    ])
    return ps, edges

def _modify_poly(poly, target, replacement=None, after=False):
    idx = list(poly.indices)
    if target in idx:
        pos = idx.index(target)
        if replacement is not None:
            idx.insert(pos + (1 if after else 0), replacement)
        else:
            idx.remove(target)
    poly.indices = [int(i) for i in idx]

def T1_left(polys, i1, i2, poly_ids, indices):
    i1, i2, i3, i4, i5, i6 = indices
    # Efficient shallow copy instead of deepcopy
    ps = [copy.copy(polys[pid]) for pid in poly_ids]
    for p in ps:
        p.indices = list(p.indices)
    
    _modify_poly(ps[0], i2)
    _modify_poly(ps[1], i1, i2)
    _modify_poly(ps[2], i1)
    _modify_poly(ps[3], i2, i1)
    
    edges = np.array([
        [i1, i2], [i2, i3], [i1, i4], [i2, i1], [i1, i5],
        [i2, i6], [i3, i2], [i4, i1], [i5, i1], [i6, i2]
    ])
    return ps, edges

def T1_right(polys, i1, i2, poly_ids, indices):
    i1, i2, i3, i4, i5, i6 = indices
    # Efficient shallow copy instead of deepcopy
    ps = [copy.copy(polys[pid]) for pid in poly_ids]
    for p in ps:
        p.indices = list(p.indices)
    
    _modify_poly(ps[0], i1)
    _modify_poly(ps[1], i1, i2, after=True)
    _modify_poly(ps[2], i2)
    _modify_poly(ps[3], i2, i1, after=True)
    
    edges = np.array([
        [i1, i2], [i1, i3], [i1, i6], [i2, i1], [i2, i5],
        [i2, i4], [i3, i1], [i4, i2], [i5, i2], [i6, i1]
    ])
    return ps, edges

def get_4_polys(polys, i1, i2):
    pids = np.full(4, -1, dtype=int)
    for p in polys:
        idx = list(p.indices)
        h1, h2 = (i1 in idx), (i2 in idx)
        if h1 and h2:
            p1, p2 = idx.index(i1), idx.index(i2)
            if (p2 - p1) % len(idx) == 1: pids[0] = p.id
            if (p1 - p2) % len(idx) == 1: pids[2] = p.id
        elif h1: pids[1] = p.id
        elif h2: pids[3] = p.id
    return pids

def T1_transition_vectorized_new(vertices, polys, edges, edgeDict, parameters):
    lx = parameters["lx"]
    ly = parameters["ly"]
    L = np.array([lx, ly])
    lmin = parameters["lmin"]
    ksep = 3

    reverse = []
    t1_ctr = 0

    edges_array = np.array(edges)
    vertices_array = np.array(vertices)

    i1_indices = edges_array[:, 0]
    i2_indices = edges_array[:, 1]

    v1_array = vertices_array[i1_indices]
    v2_array = vertices_array[i2_indices]

    v2_array = v1_array + periodic_diff(v2_array, v1_array, L)
    dist_array = np.linalg.norm(v2_array - v1_array, axis=1)

    reverse_mask = np.array([not (i1, i2) in reverse for i1, i2 in zip(i1_indices, i2_indices)])
    mask = (dist_array < lmin) & reverse_mask

    relevant_edges = np.where(mask)[0]

    if len(relevant_edges) == 0:
        return vertices_array, polys, edges, edgeDict, parameters, t1_ctr

    remove_reverse = []
    for i, edge_idx in enumerate(relevant_edges):
        i1, i2 = i1_indices[edge_idx], i2_indices[edge_idx]
        for j in range(i + 1, len(relevant_edges)):
            if [i2, i1] == [i1_indices[relevant_edges[j]], i2_indices[relevant_edges[j]]]:
                remove_reverse.append(j)
    relevant_edges = np.delete(relevant_edges, remove_reverse)

    for edge_idx in relevant_edges:
        i1 = edges[edge_idx][0]
        i2 = edges[edge_idx][1]

        if [i1, i2] in reverse:
            continue

        v1 = vertices_array[i1]
        vertex2 = vertices_array[i2]
        v2 = v1 + periodic_diff(vertex2, v1, L)

        poly_ids = get_4_polys(polys, i1, i2)

        if -1 in poly_ids:
            continue

        reverse.append([i2, i1])

        indices = get_6_indices(polys, i1, i2, poly_ids)

        polys_0, edges_0 = T1_0(polys, i1, i2, poly_ids, indices)
        E0 = get_energy(vertices_array, polys_0, edges_0, parameters)

        polys_l, edges_l = T1_left(polys, i1, i2, poly_ids, indices)
        E_left = get_energy(vertices_array, polys_l, edges_l, parameters)

        polys_r, edges_r = T1_right(polys, i1, i2, poly_ids, indices)
        E_right = get_energy(vertices_array, polys_r, edges_r, parameters)

        rC = (v1 + v2) / 2
        uv = unit_vector(v1, v2)
        perp_uv = np.array([-uv[1], uv[0]])

        v1 = rC + ksep * lmin * perp_uv
        v2 = rC - ksep * lmin * perp_uv

        vertices_array[i1] = v1
        vertices_array[i2] = v2

        rearrange_i1_i2 = check_i1_i2(vertices_array, indices, L)

        if rearrange_i1_i2:
            vertices_array[i1] = v2
            vertices_array[i2] = v1
            i1, i2 = i2, i1
            poly_ids = get_4_polys(polys, i1, i2)
            indices = get_6_indices(polys, i1, i2, poly_ids)
            polys_r, edges_r = T1_right(polys, i1, i2, poly_ids, indices)
            polys_l, edges_l = T1_left(polys, i1, i2, poly_ids, indices)
            E_left, E_right = E_right, E_left

        energy_list = [E_left, E_right]
        energy_list = np.array(energy_list)
        min_i = np.argmin(energy_list)

        if min_i == 0:
            polys, edges, n_edges, r_edges = set_T1_left(polys, polys_l, poly_ids, edges, indices)

        if min_i == 1:
            polys, edges, n_edges, r_edges = set_T1_right(polys, polys_r, poly_ids, edges, indices)

        edgeDict = get_shared_edge2(edges, polys)

        t1_ctr += 1

    return vertices_array, polys, edges, edgeDict, parameters, t1_ctr

def check_i1_i2(vertices, indices, L):
    i1 = indices[0]
    i2 = indices[1]
    i3 = indices[2]
    i4 = indices[3]
    i5 = indices[4]
    i6 = indices[5]

    v1 = vertices[i1]
    v2 = v1 + periodic_diff(vertices[i2], v1, L)
    v3 = v1 + periodic_diff(vertices[i3], v1, L)
    v4 = v1 + periodic_diff(vertices[i4], v1, L)
    v5 = v1 + periodic_diff(vertices[i5], v1, L)
    v6 = v1 + periodic_diff(vertices[i6], v1, L)

    if isInside(v1[0], v1[1], v3[0], v3[1], v6[0], v6[1], v2[0], v2[1]):
        return True
    return False

def set_T1_left(polys, polys_l, poly_ids, edges, indices):
    for i, poly in enumerate(polys_l):
        polys[poly_ids[i]].indices = poly.indices

    i1 = int(indices[0])
    i2 = int(indices[1])
    i3 = int(indices[2])
    i5 = int(indices[4])
    for i, edge in enumerate(edges):
        if edge[0] == i1 and edge[1] == i3:
            edges[i][0] = i2

        if edge[0] == i2 and edge[1] == i5:
            edges[i][0] = i1

        if edge[0] == i3 and edge[1] == i1:
            edges[i][1] = i2

        if edge[0] == i5 and edge[1] == i2:
            edges[i][1] = i1

    new_edges = np.array([[i2, i3], [i1, i5], [i3, i2], [i5, i1]]).astype(int)
    removed_edges = np.array([[i1, i3], [i2, i5], [i3, i1], [i5, i2]]).astype(int)
    return polys, edges, new_edges, removed_edges

def set_T1_right(polys, polys_r, poly_ids, edges, indices):
    for i, poly in enumerate(polys_r):
        polys[poly_ids[i]].indices = poly.indices

    i1 = int(indices[0])
    i2 = int(indices[1])
    i4 = int(indices[3])
    i6 = int(indices[5])

    for i in range(len(edges)):
        if edges[i][0] == i1 and edges[i][1] == i4:
            edges[i][0] = i2

        if edges[i][0] == i2 and edges[i][1] == i6:
            edges[i][0] = i1

        if edges[i][0] == i4 and edges[i][1] == i1:
            edges[i][1] = i2

        if edges[i][0] == i6 and edges[i][1] == i2:
            edges[i][1] = i1

    new_edges = np.array([[i2, i4], [i1, i6], [i4, i2], [i6, i1]]).astype(int)
    removed_edges = np.array([[i1, i4], [i2, i6], [i4, i1], [i6, i2]]).astype(int)

    return polys, edges, new_edges, removed_edges

# ... rest of the file remains unchanged ...
def division_transition(polys, edges, edgeDict, vertices, L, parameters, div_polys, div_times, time, rate_0, prob_0):
    cth = parameters["cth"]
    polys_0 = []
    polys_0 = polys
    if np.random.random() < prob_0:
        pick_cell_type0 = True
    else:
        pick_cell_type0 = False
    prob_0 += rate_0
    if pick_cell_type0:
        pick_poly_0 = np.random.choice(len(polys_0))
        poly = polys_0[pick_poly_0]
        if poly.cellid not in div_polys:
            div_polys.append(poly.cellid)
            div_times.append(time)
            prob_0 = 0
    removeList = []
    cellid_to_poly = {p.cellid: p for p in polys}
    for i, div_t in enumerate(div_times):
        div_flag = False
        if time - div_t > 0.1:
            div_flag = True
        target_cellid = div_polys[i]
        poly = cellid_to_poly.get(target_cellid)
        if poly is None:
            removeList.append(i)
            continue
        skip = False
        a = poly.get_area(vertices, L)
        p = poly.get_perim(vertices, L)
        c = poly.c
        if c < cth:
            skip = True
        poly_edges = get_poly_edges(poly, edges)
        verts = []
        for ind in poly.indices:
            verts.append(vertices[ind])
        if not isConvex(verts):
            skip = True
        if len(poly.indices) < 5:
            skip = True
        if a < 0.3:
            skip = True
        if div_flag and not skip:
            x, y = poly.get_center(vertices, L)
            nd, eval = poly.get_nematic_director(vertices, L)
            theta = np.arctan2(nd[1], nd[0])
            if theta < 0:
                theta -= np.pi / 2
            else:
                theta += np.pi / 2
            if (theta >= np.pi / 8 and theta < 7 * np.pi / 8) or (theta >= 9 * np.pi / 4 and theta < 17 * np.pi / 4):
                rl = 1
            else:
                rl = 0
            x1y1 = [x + L[0] * np.cos(theta), y + L[1] * np.sin(theta)]
            x2y2 = [x - L[0] * np.cos(theta), y - L[1] * np.sin(theta)]
            cut_line = [x1y1, x2y2]
            edge_lines = get_edge_lines(poly.get_poly_vertices(vertices, L))
            int_pts, int_edges = get_intersection_points(cut_line, edge_lines)
            uniques = [int_pts.index(pt) for pt in set(int_pts)]
            int_pts_list = list(set(int_pts))
            int_pts = []
            if len(int_pts_list) < 2:
                skip = True
                continue
            int_pts.append([int_pts_list[0].x, int_pts_list[0].y])
            int_pts.append([int_pts_list[1].x, int_pts_list[1].y])
            int_edges_list = int_edges
            int_edges = []
            for unique in uniques:
                int_edges.append(int_edges_list[unique])
            d_p1_e1 = distance(int_pts[0], int_edges[0])
            d_p2_e1 = distance(int_pts[1], int_edges[0])
            rearr_pts = []
            if d_p1_e1 > distance(int_pts[0], int_edges[1]): rearr_pts = [int_pts[1], int_pts[0]]
            else: rearr_pts = [int_pts[0], int_pts[1]]
            int_pts = rearr_pts[0:2]
            for pt in int_pts:
                vertices = np.append(vertices, np.reshape(pt, (1, 2)), axis=0)
            polys = change_neighbours(polys, poly.id, edges, int_edges, int_pts, vertices, rl, L)
            poly1_indices, poly2_indices = set_new_poly_indices(poly, polys, vertices, edges, L, int_pts, int_edges, rl)
            polys, edges = set_new_polys(polys, poly, poly1_indices, poly2_indices, vertices, edges, parameters)
            polys, edgeDict = renumber(polys, edgeDict, poly.id)
            cellid_to_poly = {p.cellid: p for p in polys}
            print("Divided cell ", poly.id)
            removeList.append(i)
        if skip:
            removeList.append(i)
    if len(removeList) > 0:
        for idx in sorted(removeList, reverse=True):
            del div_polys[idx]
            del div_times[idx]
    if len(removeList) > 0:
        from parser import add_reverse_edges
        edges = np.array(add_reverse_edges(np.unique(edges, axis=0).tolist()))
        edgeDict = get_shared_edge2(edges, polys)
    return polys, edges, edgeDict, vertices, div_polys, div_times, prob_0, len(removeList)

def distance(pt, edge):
    d1 = np.sqrt((pt[0] - edge[0][0]) ** 2 + (pt[1] - edge[0][1]) ** 2)
    d2 = np.sqrt((pt[0] - edge[1][0]) ** 2 + (pt[1] - edge[1][1]) ** 2)
    return min(d1, d2)

def get_edge_lines(poly_vertices):
    edge_lines = []
    n = len(poly_vertices)
    for i in range(n):
        v1 = poly_vertices[i]
        v2 = poly_vertices[(i + 1) % n]
        edge_lines.append([(v1[0], v1[1]), (v2[0], v2[1])])
    return edge_lines

def get_intersection_points(cut_line, edge_lines):
    intersection_points = []
    intersection_edges = []
    for edge_line in edge_lines:
        line1 = LineString([cut_line[0], cut_line[1]])
        line2 = LineString([edge_line[0], edge_line[1]])
        intersection = line1.intersection(line2)
        if intersection:
            intersection_points.append(intersection)
            intersection_edges.append(edge_line)
    return intersection_points, intersection_edges

def set_new_poly_indices(poly, polys, vertices, edges, L, int_pts, int_edges, rl):
    int_edges = rearrange_edges(int_edges, int_pts, vertices, rl, L)
    check_int_edges = []
    check_int_pts = []
    if rl == 1:
        if int_edges[0][1] > int_edges[2][1] or int_edges[1][1] > int_edges[3][1]:
            check_int_edges = [int_edges[2], int_edges[3], int_edges[0], int_edges[1]]
            check_int_pts = [int_pts[1], int_pts[0]]
        else:
            check_int_edges = [int_edges[0], int_edges[1], int_edges[2], int_edges[3]]
            check_int_pts = [int_pts[0], int_pts[1]]
    if rl == 0:
        if int_edges[0][0] < int_edges[2][0] or int_edges[1][0] < int_edges[3][0]:
            check_int_edges = [int_edges[2], int_edges[3], int_edges[0], int_edges[1]]
            check_int_pts = [int_pts[1], int_pts[0]]
        else:
            check_int_edges = [int_edges[0], int_edges[1], int_edges[2], int_edges[3]]
            check_int_pts = [int_pts[0], int_pts[1]]
    int_edges = check_int_edges
    int_pts = check_int_pts
    int_indices = [np.argmin(np.linalg.norm(periodic_diff(vertices, np.array(e), L), axis=1)) for e in int_edges]
    new_indices = [np.argmin(np.linalg.norm(periodic_diff(vertices, np.array(p), L), axis=1)) for p in int_pts]
    poly_indices = np.array(poly.indices)
    int_ind_0 = np.where(np.array(poly_indices) == int_indices[0])[0][0]
    int_ind_1 = np.where(np.array(poly_indices) == int_indices[1])[0][0]
    int_ind_2 = np.where(np.array(poly_indices) == int_indices[2])[0][0]
    int_ind_3 = np.where(np.array(poly_indices) == int_indices[3])[0][0]
    if not (int_ind_1 - int_ind_0) % len(poly_indices) == 1:
        int_indices[0], int_indices[1] = int_indices[1], int_indices[0]
    if not (int_ind_3 - int_ind_2) % len(poly_indices) == 1:
        int_indices[2], int_indices[3] = int_indices[3], int_indices[2]
    i1_pos = np.where(np.array(poly_indices) == int_indices[1])[0][0]
    i1_poly_indices = np.roll(poly_indices, -i1_pos)
    i3_pos = np.where(np.array(i1_poly_indices) == int_indices[3])[0][0]
    i1_poly_indices = i1_poly_indices[:i3_pos]
    mask = ~np.isin(poly_indices, i1_poly_indices)
    i4_poly_indices = poly_indices[mask]
    i4_pos = np.where(np.array(i4_poly_indices) == int_indices[3])[0][0]
    i4_poly_indices = np.roll(i4_poly_indices, -i4_pos)
    poly1_indices = np.pad(i1_poly_indices, (1, 1), "constant", constant_values=(new_indices[0], new_indices[1]))
    poly2_indices = np.pad(i4_poly_indices, (1, 1), "constant", constant_values=(new_indices[1], new_indices[0]))
    return list(poly1_indices), list(poly2_indices)

def rearrange_edges(int_edges, int_pts, vertices, rl, L):
    v1, v2, v3, v4 = int_edges[0][0], int_edges[0][1], int_edges[1][0], int_edges[1][1]
    i1, i2 = int_pts[0], int_pts[1]
    new_int_edges = []
    if rl == 1:
        if periodic_diff(v1[0], i1[0], L[0]) < 0 and periodic_diff(v2[0], i1[0], L[0]) > 0: new_int_edges.extend([v1, v2])
        else: new_int_edges.extend([v2, v1])
        if periodic_diff(v3[0], i2[0], L[0]) < 0 and periodic_diff(v4[0], i2[0], L[0]) > 0: new_int_edges.extend([v3, v4])
        else: new_int_edges.extend([v4, v3])
    elif rl == 0:
        if periodic_diff(v1[1], i1[1], L[1]) < 0 and periodic_diff(v2[1], i1[1], L[1]) > 0: new_int_edges.extend([v1, v2])
        else: new_int_edges.extend([v2, v1])
        if periodic_diff(v3[1], i2[1], L[1]) < 0 and periodic_diff(v4[1], i2[1], L[1]) > 0: new_int_edges.extend([v3, v4])
        else: new_int_edges.extend([v4, v3])
    return new_int_edges

def set_new_polys(polys, poly, poly1_indices, poly2_indices, vertices, edges, parameters):
    xi, lx, ly = parameters["xi"], parameters["lx"], parameters["ly"]
    L = np.array([lx, ly])
    max_cellid = max([p.cellid for p in polys])
    poly1 = MyPolygon(len(polys), poly1_indices, poly.A0, poly.P0, xi)
    poly2 = MyPolygon(len(polys) + 1, poly2_indices, poly.A0, poly.P0, xi)
    poly1.cellid, poly2.cellid = max_cellid + 1, max_cellid + 2
    init_chem([poly1, poly2]); poly1.cellType, poly2.cellType = poly.cellType, poly.cellType
    polys.append(poly1); polys.append(poly2)
    if poly in polys: polys.remove(poly)
    remove_indices = [i for i, edge in enumerate(edges) if edge[0] in poly.indices and edge[1] in poly.indices]
    edges = np.delete(edges, remove_indices, axis=0)
    new_edges_to_add = []
    for pi in [poly1_indices, poly2_indices]:
        for i in range(len(pi)):
            new_edges_to_add.extend([[pi[i], pi[(i+1)%len(pi)]], [pi[(i+1)%len(pi)], pi[i]]])
    if len(new_edges_to_add) > 0: edges = np.append(edges, np.array(new_edges_to_add), axis=0)
    return polys, edges

def renumber(polys, edgeDict, id):
    for poly in polys:
        if poly.id > id: poly.id -= 1
    new_dict = {}
    for key, val in edgeDict.items():
        if len(val) < 2: continue
        new_dict[key] = [x - 1 if x > id else x for x in val]
    return polys, new_dict

def change_neighbours(polys, id, edges, int_edges, int_pts, vertices, rl, L):
    int_edges = rearrange_edges(int_edges, int_pts, vertices, rl, L)
    if ((int_edges[0][1] > int_edges[2][1] or int_edges[1][1] > int_edges[3][1]) if rl == 1 else (int_edges[0][0] < int_edges[2][0] or int_edges[1][0] < int_edges[3][0])):
        int_edges, int_pts = [int_edges[2], int_edges[3], int_edges[0], int_edges[1]], [int_pts[1], int_pts[0]]
    new_indices = [np.argmin(np.linalg.norm(periodic_diff(vertices, np.array(p), L), axis=1)) for p in int_pts]
    int_idx = [np.argmin(np.linalg.norm(periodic_diff(vertices, np.array(e), L), axis=1)) for e in int_edges]
    e_inds = [[int_idx[0], int_idx[1]], [int_idx[2], int_idx[3]]]
    for p in polys:
        if p.id == id: continue
        for i, ei in enumerate(e_inds):
            if np.sum(np.isin(p.indices, ei)) == 2:
                idx0, idx1 = p.indices.index(ei[0]), p.indices.index(ei[1])
                if (idx1 - idx0) % len(p.indices) != 1 and (idx0 - idx1) % len(p.indices) != 1: continue
                local = list(ei)
                if (idx1 - idx0) % len(p.indices) != 1: local[0], local[1] = local[1], local[0]
                pos = p.indices.index(local[0])
                p.indices = [int(x) for x in p.indices[:pos+1]] + [new_indices[i]] + [int(x) for x in p.indices[pos+1:]]
    return polys

@njit(cache=True)
def shift_vertex_to_poly_frame(v, poly_verts, L):
    centroid = np.sum(poly_verts, axis=0) / len(poly_verts)
    v_shifted = v.copy()
    for dim in range(2): v_shifted[dim] += np.round((centroid[dim] - v_shifted[dim]) / L[dim]) * L[dim]
    return v_shifted

def overlap_control_new(vertices, polys, L):
    poly_vertices_list = [poly.get_poly_vertices(vertices, L) for poly in polys]
    tree = KDTree([np.mean(pv, axis=0) for pv in poly_vertices_list])
    for i in range(len(vertices)):
        v = vertices[i]
        new_vert = v.copy()
        _, idxs = tree.query(v, k=min(20, len(polys)))
        for idx in idxs:
            if i in polys[idx].indices: continue
            v_shifted = shift_vertex_to_poly_frame(new_vert, poly_vertices_list[idx], L)
            if is_point_in_polygon(v_shifted, poly_vertices_list[idx]):
                res = move_point_to_nearest_edge(v_shifted, poly_vertices_list[idx])
                if not np.isnan(res).any(): new_vert = res; break
        vertices[i] = new_vert
    return vertices

def node_switch_optimized_new(vertices, polys, edges, edgeDict, L):
    sw, new_edges, removed_edges, edges = False, [], [], list(edges)
    for _ in range(100):
        switch_made = False
        pv_list = [poly.get_poly_vertices(vertices, L) for poly in polys]
        tree, adj = KDTree([np.mean(verts, axis=0) for verts in pv_list]), [[] for _ in range(len(vertices))]
        for e in edges: adj[e[0]].append(e[1]); adj[e[1]].append(e[0])
        for i, v in enumerate(vertices):
            if switch_made: break
            _, idxs = tree.query(v, k=min(50, len(polys)))
            for idx in idxs:
                poly, poly_verts = polys[idx], pv_list[idx]
                if len(poly_verts) < 4 or i in poly.indices or not is_point_in_polygon(shift_vertex_to_poly_frame(v, poly_verts, L), poly_verts): continue
                sw = True
                if not set(adj[i]).intersection(poly.indices): overlap_control_new(vertices, polys, L); continue
                ncn = next((n for n in adj[i] if n in poly.indices), None)
                if ncn is None: continue
                other = [n for n in adj[i] if n != ncn]
                if len(other) != 2: continue
                ins = get_intersection_edge(v, vertices[other[0]], poly_verts) or get_intersection_edge(v, vertices[other[1]], poly_verts)
                if not ins or len(ins) != 2: continue
                ied = [poly.indices[ins[0] % len(poly.indices)], poly.indices[ins[1] % len(poly.indices)]]
                if ncn not in ied: continue
                dx, dy = periodic_diff(vertices[ncn][0], v[0], L[0]), periodic_diff(vertices[ncn][1], v[1], L[1])
                dim = 0 if abs(dx) > abs(dy) else 1
                cl, fa = (other[0], other[1]) if abs(periodic_diff(vertices[other[0]][dim], v[dim], L[dim])) < abs(periodic_diff(vertices[other[1]][dim], v[dim], L[dim])) else (other[1], other[0])
                nxt = ied[0] if ied[1] == ncn else ied[1]
                sh = [sp for sp in edgeDict.get(repr(np.array(sorted([ncn, i]))), []) if nxt not in polys[sp].indices]
                if not sh or cl not in polys[sh[0]].indices: cl, fa = fa, cl
                m_ed, d2 = sorted([ncn, i]), sorted([i, cl])
                mp, d2p = edgeDict.get(repr(np.array(m_ed)), []), edgeDict.get(repr(np.array(d2)), [])
                if len(mp) != 2 or len(d2p) != 2: continue
                common, only_d2, only_m = [p for p in d2p if p in mp], [p for p in d2p if p not in mp], [p for p in mp if p not in d2p]
                if not common or not only_d2 or not only_m: continue
                pf = (0 if nxt == poly.indices[(poly.indices.index(ncn) - 1) % len(poly.indices)] else 1)
                poly.indices.insert(poly.indices.index(ncn) + pf, i)
                polys[only_d2[0]].indices.insert(polys[only_d2[0]].indices.index(i) + pf, ncn)
                for px in [only_m[0], common[0]]: polys[px].indices = [vi for vi in polys[px].indices if vi != (ncn if px == only_m[0] else i)]
                for de in [sorted([ncn, nxt]), d2]:
                    for dir in [de, de[::-1]]:
                        try: edges.remove(dir); removed_edges.append(dir)
                        except ValueError: pass
                for ae in [[i, nxt], [ncn, cl]]:
                    for ed in [ae, ae[::-1]]: edges.append(ed); new_edges.append(ed)
                edgeDict, switch_made = get_shared_edge2(np.array(edges), polys), True; break
        if not switch_made: break
    return polys, np.array(edges), edgeDict, sw, np.array(new_edges), np.array(removed_edges)
