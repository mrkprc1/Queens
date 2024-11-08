import cv2 as cv
import numpy as np
import gradio as gr
from utils import (
    test_approx_quadri,
    contour_points_to_lines,
    plot_img_and_contours,
    plot_thresh_and_contours,
    find_queens_grid, 
    plot_queens_to_img, 
    solve_queens)

def run_queens_solver(img_path, debug=False):

    # Load the image.
    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    # Find the game board and extract the colours.
    queens_array, cmap = find_queens_grid(img, debug=debug)

    # If the game board was not found, return a "no Queens Game found image".
    if queens_array is None:
        return cv.imread('NoQueens.png')

    # Solve the game.
    queens_solved = solve_queens(queens_array, cmap)
    
    return plot_queens_to_img(queens_array, cmap, queens=np.argwhere(queens_solved == 1).tolist())


demo = gr.Interface(
    fn=run_queens_solver,
    inputs=gr.Image(type='filepath', label="Image of the Queens game board"),
    outputs=gr.Image()
)

demo.launch(share=False)
