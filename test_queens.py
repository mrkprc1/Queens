import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from scipy.optimize import LinearConstraint, Bounds, milp
from utils import (
    test_approx_quadri,
    contour_points_to_lines,
    plot_img_and_contours,
    plot_thresh_and_contours,
    find_queens_grid, 
    plot_queens)

# Load the image.
img_files = os.listdir('QueensScreenshots')
for img_file in img_files:
    if img_file.endswith('.png'):
        img = cv.imread('QueensScreenshots/' + img_file)
# img = cv.imread('QueensScreenshots/Screenshot 2024-11-06 at 08.31.05.png')

    # Find the game board and extract the colours.
    queens_array, cmap = find_queens_grid(img, debug=False)

    # Plot the game board.
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plot_queens(queens_array, cmap)

    # Define the optimsation problem.
    n = queens_array.shape[0]
    num_bits = n**2
    id_n = np.identity(n)
    t = []
    for i in range(n-1):
        for j in range(n-1):
            t += [np.pad(np.ones((2,2), dtype=int), ((j,n-2-j),(i, n-2-i))).ravel()]
    A_surround = np.array(t)
    A_row_col = np.concatenate([
        np.kron(id_n, np.ones((1, n))),
        np.kron(np.ones((1, n)), id_n)
    ])
    A_region = np.zeros((n, n**2), dtype=int)
    for i in range(n):
        region_idx = np.where(queens_array == i)
        idx = [n*x+region_idx[1][i] for i,x in enumerate(region_idx[0])]
        A_region[i, idx] = 1

    # Set the objective as zero, since we are only interested in finding a feasible solution.
    c = np.zeros(num_bits, dtype=np.int64)

    # Set the bounds as lb=0, ub=1 for all variables to ensure they are binary.
    lb = np.zeros(num_bits, dtype=np.int64)
    ub = np.ones(num_bits, dtype=np.int64)

    # Define the linear constraints and bounds objects.
    lin_con_row_col = LinearConstraint(A_row_col, np.ones(A_row_col.shape[0], dtype=np.int64), np.ones(A_row_col.shape[0], dtype=np.int64))
    lin_con_surround = LinearConstraint(A_surround, np.zeros(A_surround.shape[0], dtype=np.int64), np.ones(A_surround.shape[0], dtype=np.int64))
    lin_con_region = LinearConstraint(A_region, np.ones(A_region.shape[0], dtype=np.int64), np.ones(A_region.shape[0], dtype=np.int64))
    constraints = [lin_con_row_col, lin_con_region, lin_con_surround]
    bounds = Bounds(lb, ub)

    # Solve the MILP.
    res = milp(c, integrality=np.ones(num_bits), constraints=constraints, bounds=bounds)

    plot_queens(queens_array, cmap, queens=[(i//n, i%n) for i in range(num_bits) if res.x[i] == 1])