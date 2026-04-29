import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from geometry import periodic_diff


def boundary_cell(poly, vertices, L):
    i = poly.indices[0]
    for j in poly.indices:
        if np.linalg.norm(vertices[j] - vertices[i]) > 0.5 * L[0]:
            return True
    return False


def eight_disp(vertex, L):
    x, y = vertex
    return [
        [x, y],
        [x + L[0], y],
        [x - L[0], y],
        [x, y + L[1]],
        [x, y - L[1]],
        [x + L[0], y + L[1]],
        [x - L[0], y - L[1]],
        [x + L[0], y - L[1]],
        [x - L[0], y + L[1]],
    ]


def boundary_vertices(poly, vertices, L):
    verts = []
    i = poly.indices[0]
    for j in poly.indices:
        if np.linalg.norm(vertices[j] - vertices[i]) < 0.5 * L[0]:
            verts.append(vertices[j])
        else:
            for v in eight_disp(vertices[j], L):
                if np.linalg.norm(vertices[i] - v) < 0.5 * L[0]:
                    verts.append(v)
    return verts


def plot_tissue(vertices, polys, L, file):
    plt.cla()
    fig = plt.figure(figsize=(L[0], L[1]))
    ax = fig.add_subplot(1, 1, 1)

    col = ["lightgrey", "#50C878"]
    patches = []

    for poly in polys:
        indices = poly.indices
        c = col[poly.cellType]

        if boundary_cell(poly, vertices, L):
            new_verts = boundary_vertices(poly, vertices, L)
            patches.append(
                Polygon(
                    new_verts,
                    closed=True,
                    facecolor=c,
                    edgecolor=(1, 1, 1, 1),
                    linewidth=1,
                )
            )
            span_verts = np.zeros((8, len(new_verts), 2))
            for j, nv in enumerate(new_verts):
                disp_nv = eight_disp(nv, L)
                for i in range(8):
                    span_verts[i, j] = disp_nv[i]
            for i in range(8):
                patches.append(
                    Polygon(
                        span_verts[i, :],
                        closed=True,
                        facecolor=c,
                        edgecolor=(1, 1, 1, 1),
                        linewidth=1,
                    )
                )
        else:
            patches.append(
                Polygon(
                    vertices[indices],
                    closed=True,
                    facecolor=c,
                    edgecolor=(1, 1, 1, 1),
                    linewidth=1,
                )
            )

        for i, index in enumerate(indices):
            x1, y1 = vertices[index]
            if i == len(indices) - 1:
                x2, y2 = vertices[indices[0]]
            else:
                x2, y2 = vertices[indices[i + 1]]

            v1 = np.array((x1, y1))
            v2 = np.array((x2, y2))
            v2 = v1 + periodic_diff(v2, v1, L)
            x2, y2 = v2
            ax.plot([x1, x2], [y1, y2], c="k", ls="-", lw=1.5)

            v2 = np.array((x2, y2))
            v1 = v2 + periodic_diff(v1, v2, L)
            x1, y1 = v1
            ax.plot([x1, x2], [y1, y2], c="k", ls="-", lw=1.5)

        x, y = poly.get_center(vertices, L)
        poly.get_nematic_director(vertices, L)
        ax.quiver(
            x,
            y,
            poly.nd[0],
            poly.nd[1],
            color="k",
            pivot="mid",
            headlength=0,
            headwidth=1,
            linewidth=0.1,   # edge line thickness
            width=0.003,    # shaft thickness (main control)
            scale=50,
            zorder=1000,
        )

    patch_collection = PatchCollection(patches, match_original=True)
    ax.add_collection(patch_collection)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axis([0, L[0], 0, L[1]])

    plt.savefig(file, bbox_inches="tight", dpi=100)
    plt.close(fig)
