#!/usr/bin/python
import numpy as np

def trace_periodic_vertices(nx, ny, xx, yy, w, h, L):
    """
    Traces hexagonal vertices across a periodic grid efficiently.
    
    Args:
        nx, ny: Number of hexagons in X and Y directions.
        xx, yy: Coordinate matrices from grid generation.
        w, h: Unit dimensions (unused, kept for compatibility).
        L: Box dimensions (unused, kept for compatibility).
        
    Returns:
        vertices: Array of unique (x, y) coordinates.
        hex_indices: (N_hex, 6) array mapping hexagons to vertex indices.
    """
    len_y, len_x = xx.shape
    n_hex = nx * ny
    
    vertex_map = {}
    unique_vertices = []
    
    def get_vertex_id(ix, iy):
        px, py = ix % len_x, iy % len_y
        vx, vy = xx[py, px], yy[py, px]
        
        coord = (float(vx), float(vy))
        if coord not in vertex_map:
            vertex_map[coord] = len(unique_vertices)
            unique_vertices.append([vx, vy])
        return vertex_map[coord]

    offsets = [(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)]
    
    hex_indices = np.zeros((n_hex, 6), dtype=int)
    hex_count = 0
    
    x_starts = np.arange(0, len_x, 3)
    x_left_0 = [x for x in x_starts if x % 2 != 0]
    x_left_1 = [x for x in x_starts if x % 2 == 0]
    
    for iy in range(2 * ny):
        current_starts = x_left_0 if iy % 2 == 0 else x_left_1
        
        for ix in current_starts:
            for i, (dx, dy) in enumerate(offsets):
                hex_indices[hex_count, i] = get_vertex_id(ix + dx, iy + dy)
            hex_count += 1
            
    vertices = np.array(unique_vertices)
    
    expected_n_v = 2 * n_hex
    if len(vertices) < expected_n_v:
        padding = np.full((expected_n_v - len(vertices), 2), -1.0)
        vertices = np.vstack([vertices, padding])
        
    return vertices, hex_indices
