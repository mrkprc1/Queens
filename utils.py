from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties

def plot_thresh_and_contours(thresh, contours, thickness=3):
    cont = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype='uint8')
    max_x = thresh.shape[0]-1
    max_y = thresh.shape[1]-1
    for contour in contours:
        for i in contour:
            for t in range(thickness+1):
                cont[min(i[0][1]+t, max_x), min(i[0][0]+t, max_y), 1] = 255
                cont[max(i[0][1]-t, 0), max(i[0][0]-t, 0), 1] = 255
                cont[min(i[0][1]+t, max_x), max(i[0][0]-t, 0), 1] = 255
                cont[max(i[0][1]-t, 0), min(i[0][0]+t, max_y), 1] = 255

    temp = np.tile(thresh.T, (3, 1, 1)).T.copy()
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], np.zeros(len(np.where(cont==255)[1]), dtype='uint8'))] = 0
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], np.ones(len(np.where(cont==255)[1]), dtype='uint8'))] = 0
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], 2*np.ones(len(np.where(cont==255)[1]), dtype='uint8'))] = 0

    plt.imshow(temp + cont)
    plt.show()
    return

def plot_img_and_contours(img, contours, thickness=3, channel=1):
    cont = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    max_x = img.shape[0]-1
    max_y = img.shape[1]-1
    for contour in contours:
        for i, p1 in enumerate(np.squeeze(contour)):
            for t in np.arange(-thickness, thickness+1):
                s = np.sign(t)
                if s==1:
                    idx = min(p1[1]+t, max_x)
                elif s==-1:
                    idx = max(p1[1]+t, 0)
                else:
                    idx = p1[1]
                for t1 in np.arange(-thickness, thickness+1):
                    s1 = np.sign(t1)
                    if s1==1:
                        idy = min(p1[0]+t1, max_y)
                    elif s1==-1:
                        idy = max(p1[0]+t1, 0)
                    else:
                        idy = p1[0]
                    if t**2 + t1**2 > thickness**2:
                        continue
                    else:
                        cont[idx, idy, channel] = 255
                    # cont[max(i[0][1]-t, 0), max(i[0][0]-t, 0), 1] = 255
                    # cont[min(i[0][1]+t, max_x), max(i[0][0]-t, 0), 1] = 255
                    # cont[max(i[0][1]-t, 0), min(i[0][0]+t, max_y), 1] = 255

    if len(img.shape) == 2:
        temp = np.tile(img.T, (3, 1, 1)).T.copy()
    else:
        temp = img.copy()
    # temp = np.tile(img.T, (3, 1, 1)).T.copy()
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], np.zeros(len(np.where(cont==255)[1]), dtype='uint8'))] = 0
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], np.ones(len(np.where(cont==255)[1]), dtype='uint8'))] = 0
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], 2*np.ones(len(np.where(cont==255)[1]), dtype='uint8'))] = 0

    plt.imshow(temp + cont)
    plt.show()
    return

def plot_thresh_and_moments(thresh, moments, thickness=3):
    cont = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype='uint8')
    max_x = thresh.shape[0]-1
    max_y = thresh.shape[1]-1
    print(max_x, max_y)
    for moment in moments:
        for t in range(thickness+1):
            cont[min(moment[1]+t, max_x), min(moment[0]+t, max_y), 1] = 255
            cont[max(moment[1]-t, 0), max(moment[0]-t, 0), 1] = 255
            cont[min(moment[1]+t, max_x), max(moment[0]-t, 0), 1] = 255
            cont[max(moment[1]-t, 0), min(moment[0]+t, max_y), 1] = 255

    temp = np.tile(thresh.T, (3, 1, 1)).T.copy()
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], np.zeros(len(np.where(cont==255)[1]), dtype='uint8'))] = 0
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], np.ones(len(np.where(cont==255)[1]), dtype='uint8'))] = 0
    temp[(np.where(cont==255)[0], np.where(cont==255)[1], 2*np.ones(len(np.where(cont==255)[1]), dtype='uint8'))] = 0

    plt.imshow(temp + cont)
    plt.show()
    return

def contour_points_to_lines(contour):
    lines = []
    for i,p1 in enumerate(np.squeeze(contour)):
        if i==len(contour)-1:
            p2 = contour[0][0]
        else:
            p2 = contour[i+1][0]
        dim_max = np.argmax(np.abs(p2-p1))
        max_dir = np.sign((p2-p1)[dim_max])
        max_range = np.arange(p1[dim_max], p2[dim_max]+max_dir, max_dir)
        s_vec = (max_range - p1[dim_max]) / (p2[dim_max] - p1[dim_max])
        dim_min = (dim_max+1)%2
        if dim_max==0:
            lines += [np.array([[[max_range[i], (p1[dim_min] + s*(p2[dim_min] - p1[dim_min])).astype(int)]] for i,s in enumerate(s_vec)])]
        else:
            lines += [np.array([[[(p1[dim_min] + s*(p2[dim_min] - p1[dim_min])).astype(int), max_range[i]]] for i,s in enumerate(s_vec)])]
        # lines += [np.array([[[max_range[i], (p1[dim_min] + s*(p2[dim_min] - p1[dim_min])).astype(int)]] for i,s in enumerate(s_vec)])]
    return np.concatenate(lines)

def test_approx_quadri(contour, threshold=0.01):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, threshold * peri, True)
    # We are looking for a contour that is roughly a quadrilateral
    if len(approx) == 4:
        return True
    else:
        return False
    

def find_queens_grid(img, debug=False):
    # Threshold image and find contours.
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Get contours that are approximately quadrilateral, sorted by perimeter.
    quadri_idx = [i for i, c in enumerate(contours) if test_approx_quadri(c, 0.1)]
    quad_contours = [contours[i] for i in quadri_idx]
    quad_hierarchy = hierarchy[:, quadri_idx, :]

    areas = [cv.contourArea(c, True) for c in quad_contours]
    idxs = list(np.argsort(areas))
    idxs.reverse()
    quad_contours = [quad_contours[i] for i in idxs]
    quad_hierarchy = quad_hierarchy[:, idxs, :]

    # Find the grid of the game (should be the biggest quadrilateral).
    game_grid = quad_contours[0]

    # Crop the image to the game grid.
    x,y,w,h = cv.boundingRect(game_grid)
    cropped_thresh = thresh[y:y+h, x:x+w]
    cropped_img = img[y:y+h, x:x+w]
    if debug:
        bbox = np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
        plot_img_and_contours(img, [contour_points_to_lines(con) for con in [bbox]], thickness=1)

        plt.imshow(cropped_img)
        plt.axis('off')
        plt.show()

    # Find the contours inside of the game grid.
    contours,hierarchy = cv.findContours(cropped_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    quadri_idx = [i for i, c in enumerate(contours) if test_approx_quadri(c, 0.1)]
    quad_contours = [contours[i] for i in quadri_idx]
    quad_hierarchy = hierarchy[:, quadri_idx, :]

    # Order the contours by area.
    areas = [cv.contourArea(c) for c in quad_contours]
    idxs = list(np.argsort(areas))
    idxs.reverse()
    quad_hierarchy = quad_hierarchy[:, idxs, :]

    # Look for the cells of the game grid. These will be the quadrilaterals
    # that are closest to the area of the game grid divided by the number of 
    # cells, which must be a square number. 
    tol = 0.3
    for i in range(4, 11):
        cell_area = cv.contourArea(game_grid) / i**2
        lb = cell_area*(1-tol)
        ub = cell_area
        cells_idxs = [idx for idx in idxs if areas[idx]<=ub and areas[idx]>=lb]

        if len(cells_idxs) == i**2:
            n = i
            cells = [quad_contours[idx] for idx in cells_idxs]
            break

    if cells is None:
        print("Could not find the cells.")

    if debug:
        plot_img_and_contours(cropped_img, [contour_points_to_lines(con) for con in cells], thickness=1)

    # Get moments of the cells and find the centroids.
    cs=[]
    for cell in cells:
        m = cv.moments(cell)
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
        cs += [[cx, cy]]
    cs = np.array(cs)

    # Determine the indices of the cells.
    idx_array = np.empty((n,n))
    cells_ordered = []
    for i in range(n):
        row_i = np.argsort(cs[:,1])[i*n:(i+1)*n]
        cols = row_i[np.argsort(cs[row_i][:,0])]
        idx_array[i] = cols
        cells_ordered += [cells[col] for col in cols]
    
    if debug:
        plot_thresh_and_contours(cropped_thresh, [contour_points_to_lines(cell) for cell in cells_ordered[0:n]], thickness=0)

    # Get the colours of the cells.
    colour_array = np.zeros((n,n,3), dtype=int)
    for i, cell in enumerate(cells_ordered):
        row = i//n
        col = i%n
        x,y,w,h = cv.boundingRect(cell)
        cropped_cell = cropped_img[y:y+h, x:x+w]
        colour_array[row, col, :] = np.median(cropped_cell, axis=(0,1)).astype(int)

    # Number the regions of the board 0 to n-1 and get the mapping
    # from the colours to the regions.
    numbered_array = np.zeros((n,n), dtype=int)
    colour_order = []
    for i, cell in enumerate(cells_ordered):
        row = i//n
        col = i%n
        colour = colour_array[row, col]
        if len(colour_order) == 0:
            colour_order += [colour]
            region = 0
            continue
        if not np.any(np.all(colour == colour_order, axis=1)):
            colour_order += [colour]
            region = len(colour_order) - 1
        else:
            region = np.where(np.all(colour == colour_order, axis=1))[0][0]
        numbered_array[row, col] = region
    cmap = ListedColormap([col/255 for col in colour_order])

    return numbered_array, cmap


def plot_queens(numbered_array, cmap, queens=None):
    # Load the Font Awesome font for the Moon and Sun symbols.
    prop = FontProperties(fname='Font Awesome 6 Free-Solid-900.otf')

    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(numbered_array, cmap=cmap)
    n = numbered_array.shape[0]
    for i in range(n+1):
        ax.axhline(y=i-0.5, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=i-0.5, color='black', linestyle='-', linewidth=2)
    plt.axis('off')
    if queens is not None:
        for queen in queens:
            
            ax.annotate('\uf521', [queen[1], queen[0]], color='black', ha='center', va='center', fontsize=20, fontproperties=prop)

    plt.show()