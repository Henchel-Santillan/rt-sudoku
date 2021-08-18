import cv2
import numpy as np


KERNEL_SIZE = (5, 5)
POLY_APPROX_COEFFICIENT = 0.015


def preprocess(image):
    """
    Preprocessing the image involves
    1. Segmentation of the image via thresholding
    2. Detecting the "blob" of the puzzle by assuming it to be the largest in the image sample
    3. Applying hough transform to locate the four corners of the puzzle
    4. Remapping the sample image to a resized image that fits corner to corner (downsampling)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, KERNEL_SIZE, 0)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)

    return threshold


def order_corners(corners):
    """
    N.B.: Index 0 = top-right
                1 = top-left
                2 = bottom-left
                3 = bottom-right
    """
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    return corners[1], corners[0], corners[3], corners[2]


def perspective_transform(image, corners):
    ordered = order_corners(corners)
    top_l, top_r, bot_r, bot_l = ordered    # 4-tuple unpacking

    # width of new image is the maximum distance between bot_l and bot_r or top_l and top_r
    width_bot = np.sqrt((bot_r[0] - bot_l[0]) ** 2 + (bot_r[1] - bot_l[1]) ** 2)
    width_top = np.sqrt((top_r[0] - top_l[0]) ** 2 + (top_r[1] - top_l[1]) ** 2)
    width = max(int(width_bot), int(width_top))

    # repeat the process for the new image height
    height_l = np.sqrt((top_r[0] - bot_r[0]) ** 2 + (top_r[1] - bot_r[1]) ** 2)
    height_r = np.sqrt((top_l[0] - bot_l[0]) ** 2 + (top_l[1] - bot_l[1]) ** 2)
    height = max(int(height_l), int(height_r))

    # construct an np array with top-down view in order
    dims = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    ordered = np.array(ordered, dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, dims)

    return cv2.warpPerspective(image, matrix, (width, height))


def find_contours(src, image_p):
    kernel = np.ones((3, 3), dtype="uint8")

    kernel[0][0] = 0
    kernel[0][2] = 0
    kernel[2][0] = 0
    kernel[2][2] = 0

    dilated_image = cv2.dilate(image_p, kernel)
    contours, _ = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_p, max_c = 0, None

    for c in contours:
        p = cv2.arcLength(c, True)
        poly_approx = cv2.approxPolyDP(c, POLY_APPROX_COEFFICIENT * p, True)
        if len(poly_approx) == 4 and p > max_p:
            max_p = p
            max_c = poly_approx

    return perspective_transform(src, max_c)


def find_lines(image_t):
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


sample = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg')
transformed = find_contours(sample, preprocess(sample))
cv2.imshow("", find_lines(transformed))
cv2.waitKey(0)

