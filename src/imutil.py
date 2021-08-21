import os
import shutil

import cv2
import numpy as np


KERNEL_SIZE = (5, 5)
POLY_APPROX_COEFFICIENT = 0.015
ARB_RHO_THRESH = 15
ARB_THETA_THRESH = 0.1
DIM = 9
PATH_TO_TEMP = r"C:\Users\hench\PycharmProjects\rt-sudoku\temp"


def preprocess(image):
    """
    Image preprocessing: greyscale, low-pass filter via Gaussian blur, adaptive thresholding
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, KERNEL_SIZE, 0)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    return threshold


def order_corners(corners):
    """
    Orders the corners in clockwise order beginning at the top right
    """

    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    return corners[1], corners[0], corners[3], corners[2]


def perspective_transform(image, corners):
    """
    Four-point transform, obtains a consistent order of the points and unpacks them individually
    """

    ordered = order_corners(corners)
    top_l, top_r, bot_r, bot_l = ordered  # 4-tuple unpacking

    # width of new image is the maximum distance between bot_l and bot_r or top_l and top_r
    width_bot = np.sqrt((bot_r[0] - bot_l[0]) ** 2 + (bot_r[1] - bot_l[1]) ** 2)
    width_top = np.sqrt((top_r[0] - top_l[0]) ** 2 + (top_r[1] - top_l[1]) ** 2)
    width = max(int(width_bot), int(width_top))

    # repeat the process for the new image height
    height_l = np.sqrt((top_r[0] - bot_r[0]) ** 2 + (top_r[1] - bot_r[1]) ** 2)
    height_r = np.sqrt((top_l[0] - bot_l[0]) ** 2 + (top_l[1] - bot_l[1]) ** 2)
    height = max(int(height_l), int(height_r))

    # construct an np array with top-down view in order
    dims = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    ordered = np.array(ordered, np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dims)

    return cv2.warpPerspective(image, matrix, (width, height))


def find_contours(src, image_p):
    kernel = np.ones((3, 3), np.uint8)

    kernel[0][0] = 0
    kernel[0][2] = 0
    kernel[2][0] = 0
    kernel[2][2] = 0

    dilated_image = cv2.dilate(image_p, kernel)
    cs, _ = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)

    max_p, max_c = 0, None

    for c in cs:
        p = cv2.arcLength(c, True)
        poly_approx = cv2.approxPolyDP(c, POLY_APPROX_COEFFICIENT * p, True)
        if len(poly_approx) == 4 and p > max_p:
            max_p = p
            max_c = poly_approx

    if max_c is not None:
        return perspective_transform(src, max_c)
    return src


def find_lines(image_t):
    """
    Uses the standard Hough transform + filtering to detect the grid lines of the puzzle
    """

    gray = cv2.cvtColor(image_t, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125)

    for line in lines:
        rho, theta = line[0]
        h, v = np.cos(theta), np.sin(theta)
        h_rho, v_rho = h * rho, v * rho
        x1, y1 = int(h_rho + 1000 * (-v)), int(v_rho + 1000 * h)
        x2, y2 = int(h_rho - 1000 * (-v)), int(v_rho - 1000 * h)

        cv2.line(image_t, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image_t


def find_grid(image_p):
    image = cv2.bitwise_not(image_p)
    cs, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_max = max(cs, key=cv2.contourArea)

    x, y, width, height = cv2.boundingRect(c_max)
    grid = image[y: y + height, x: x + width]
    grid = cv2.resize(grid, (min(grid.shape), min(grid.shape)))

    cs2, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cs2 = sorted(cs2, key=cv2.contourArea, reverse=True)

    maximum = None
    poly_approx = None

    for c in cs2[:min(5, len(cs2))]:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, POLY_APPROX_COEFFICIENT * perimeter, True)
        if len(approx) == 4:
            maximum = c
            poly_approx = approx

    if maximum is not None:
        grid = perspective_transform(grid, poly_approx)

    return grid


def extract_cells(image_t):
    # Clear temp folder first
    for f in os.listdir(PATH_TO_TEMP):
        os.remove(os.path.join(PATH_TO_TEMP, f))

    '''
    for fname in os.listdir(PATH_TO_TEMP):
        try:
            if os.path.isfile(fname) or os.path.islink(fname):
                os.unlink(fname)
            elif os.path.isdir(fname):
                shutil.rmtree(fname)
        except Exception as e:
            print("Failed to remove file %s. See below: %s" % (fname, e))
    '''

    image = image_t.copy()
    width, height = image.shape

    image = cv2.resize(image, (min(width, height), min(width, height)))
    cell_dim = int(width / DIM)

    if cell_dim != 0:
        i, j = 0, 0
        for r in range(0, width - cell_dim, cell_dim):
            j = 0
            for c in range(0, height - cell_dim, cell_dim):
                f = DIM + 1
                cell = image[r + f: r + cell_dim - f, c + f: c + cell_dim - f]
                fname = "cell" + str(i) + str(j) + ".png"
                cv2.imwrite(os.path.join(PATH_TO_TEMP, fname), cell)
                j += 1
            i += 1


sample = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg')
transformed = find_contours(sample, preprocess(sample))
# extract_cells(preprocess(transformed))

# cv2.imshow("", find_grid(preprocess(transformed)))
cv2.waitKey(0)
cv2.destroyAllWindows()
