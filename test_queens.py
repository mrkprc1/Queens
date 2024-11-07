import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
import pytest
from scipy.optimize import LinearConstraint, Bounds, milp
from utils import (
    find_queens_grid, 
    solve_queens)

queens_games = pickle.load(open('queens_games.pkl', 'rb'))

@pytest.mark.parametrize('game_data', queens_games)
def test_find_queens_grid(game_data):
    # Load the image.
    img_file = game_data['img']
    img = cv.cvtColor(cv.imread(f'QueensScreenshots/{img_file}'), cv.COLOR_BGR2RGB)
    
    # Find the game board and extract the colours.
    queens_array, cmap = find_queens_grid(img, debug=False)
    
    # Plot the game board.
    assert np.all(queens_array == game_data['queens_array'])
    assert np.allclose(cmap.colors, game_data['cmap'])

@pytest.mark.parametrize('game_data', queens_games)
def test_queens_solver(game_data):
    # Load the extracted game data.
    queens_array = game_data['queens_array']
    cmap = ListedColormap(game_data['cmap'])

    # Run solver.
    queens_array_solved = solve_queens(queens_array, cmap)

    # Check the solution.
    assert np.all(queens_array_solved == game_data['queens_array_solved'])

