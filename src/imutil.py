import os

import cv2
import numpy as np


KERNEL_SIZE = (5, 5)
POLY_APPROX_COEFFICIENT = 0.015
ARB_RHO_THRESH = 15
ARB_THETA_THRESH = 0.1
DIM = 9
PATH_TO_TEMP = r"C:\Users\hench\PycharmProjects\rt-sudoku\temp"


skewed = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_skewed.jpg')
sample = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg')
perfect = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_perfect.png')


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


def order_corners2(points):
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of (x, y) for each corner point: min = top-left, max = bottom-right
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # Difference of (x, y): min = top-right, max = bottom-left
    d = np.diff(points, axis=1)
    rect[1] = points[np.argmin(d)]
    rect[3] = points[np.argmax(d)]

    return rect


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
    dims = np.array([[0, 0],
                     [width - 1, 0],
                     [width - 1, height - 1],
                     [0, height - 1]],
                    np.float32)

    ordered = np.array(ordered, np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dims)

    return cv2.warpPerspective(image, matrix, (width, height))


def skew_angle(image):
    image_p = preprocess(image)
    image_p = cv2.bitwise_not(image_p)
    dilated = cv2.dilate(image_p, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5)))

    cs, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)

    min_rect = cv2.minAreaRect(cs[0])
    angle = min_rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotate_image(image, angle: float):
    image_c = image.copy()
    h, w = image_c.shape[:2]
    centre = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    image_c = cv2.warpAffine(image_c, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image_c


def fix_image_skew(image):
    angle = skew_angle(image)
    return rotate_image(image, -1.0 * angle)


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

        cv2.line(image_t, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_t


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def partition(image):
    """
    Manual partition algorithm. Should break image into N x N squares
    """
    for f in os.listdir(PATH_TO_TEMP):
        os.remove(os.path.join(PATH_TO_TEMP, f))

    h, w = image.shape[:2]
    resized = cv2.resize(image.copy(), (min(w, h), min(w, h)))

    rdim = resized.shape[:2][0]
    box_dim = int(rdim / DIM)
    if box_dim != 0:
        x, y = 1, 1
        for r in range(0, rdim - box_dim, box_dim):
            for c in range(0, rdim - box_dim, box_dim):
                cell = resized[r: r + box_dim, c: c + box_dim]
                fname = str(x) + str(y) + ".png"
                cv2.imwrite(os.path.join(PATH_TO_TEMP, fname), cell)
                y += 1
            y = 0
            x += 1
