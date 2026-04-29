# Code to plot cell pressure

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from geometry import *
from Polygon import Polygon
from plot import plot_tissue
import sys

foldername = sys.argv[1]

T = np.arange(0, 1000.01, 1)

for t in tqdm(T):
    data_file = os.path.join(foldername, f"raw/data_{t:.2f}.npz")
    save_folder = os.path.join(foldername, "tissue")

    if not os.path.exists(data_file):
        print(f"Data file {data_file} does not exist. Skipping.")
        continue

    data = np.load(data_file, allow_pickle=True)
    vertices = data['vertices']
    polys = data['polys']
    edges = data['edges']
    L = data['L']

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plot_tissue(vertices, polys, L, os.path.join(save_folder, f"{t:07.2f}.jpg"))