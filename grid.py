#!/usr/bin/python
import numpy as np

"""
Grid generation logic for hexagonal grids.
Spacing principles:
- X-axis step: 0.25 * width (w)
- Y-axis step: 0.5 * height (h)
- Horizontal scale: 1.5 * w (per nx block)
- Vertical scale: 1.0 * h (per ny block)
"""

def generate_periodic_grid(nx, ny, w, h):
    """
    Generates a periodic hexagonal coordinate grid.
    
    Args:
        nx: Number of cells in x direction.
        ny: Number of cells in y direction.
        w: Unit width.
        h: Unit height.
        
    Returns:
        xx, yy: Coordinate matrices.
        L: Box dimensions [lx, ly].
    """
    dx = 0.25 * w
    dy = 0.5 * h
    
    # Total side lengths based on 3 steps per nx and 2 steps per ny
    lx = 3 * nx * dx
    ly = 2 * ny * dy
    
    # Create periodic axes by excluding the endpoint
    xs = np.linspace(0., lx, 3 * nx, endpoint=False)
    ys = np.linspace(0., ly, 2 * ny, endpoint=False)
    
    xx, yy = np.meshgrid(xs, ys)
    L = np.array([lx, ly])
    
    return xx, yy, L

def generate_periodic_grid_diff(nx, ny, numx, numy, w, h):
    """
    Maintains compatibility with the original signature while using optimized logic.
    Note: numx and numy were unused in the original implementation.
    """
    return generate_periodic_grid(nx, ny, w, h)

def generate_grid(nx, ny, w, h):
    """
    Generates a non-periodic hexagonal coordinate grid.
    Includes the boundary points.
    """
    dx = 0.25 * w
    dy = 0.5 * h
    
    # Non-periodic bounds include an extra partial step
    nx_steps = 3 * nx + 1
    ny_steps = 2 * ny + 1
    
    lx = nx_steps * dx
    ly = ny_steps * dy
    
    # Generate coordinates including endpoints
    xs = np.linspace(0., lx, nx_steps + 1)
    ys = np.linspace(0., ly, ny_steps + 1)
    
    xx, yy = np.meshgrid(xs, ys)
    L = np.array([lx, ly])
    
    return xx, yy, L
