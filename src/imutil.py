import cv2
import numpy as np

'''
KERNEL_SIZE = 9
CONTOUR_MAX = 4  # 4 for 4 sides
CONTOUR_AREA_MIN = 10000


def preprocess(img):
    # N.B. img arg is in greyscale
    # Gaussian Blur + Adaptive Thresholding
    improc = cv2.GaussianBlur(img.copy(), (KERNEL_SIZE, KERNEL_SIZE), 0)
    improc = cv2.adaptiveThreshold(improc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Color inversion and dilation to remove noise
    improc = cv2.bitwise_not(improc, improc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    improc = cv2.dilate(improc, kernel)
    return improc


def find_contours(thresh):
    count = 0
    max_area = -1
    max_contour = None

    height, width, _ = thresh.shape

    for y in range(height):
'''

KERNEL_SIZE = (5, 5)
POLY_APPROX_COEFFICIENT = 0.02

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

    threshold = cv2.bitwise_not(threshold, threshold)
    return threshold
    # cv2.Canny(threshold, 100, 200)


def detect_grid(image_p):
    pass


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
    width_bot = np.sqrt(((bot_r[0] - bot_l[0]) ** 2) + ((bot_r[1] - bot_l[1]) ** 2))
    width_top = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_bot), int(width_top))

    # repeat the process for the new image height
    height_l = np.sqrt(((top_r[0] - bot_r[0]) ** 2) + ((top_r[1] - bot_r[1]) ** 2))
    height_r = np.sqrt(((top_l[0] - bot_l[0]) ** 2) + ((top_l[1] - bot_l[1]) ** 2))
    height = max(int(height_l), int(height_r))

    # construct an np array with top-down view in order
    ordered = np.array(ordered, dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]))
    return cv2.warpPerspective(image, matrix, (width, height))


def warp_with_contours(image_p):
    contours = cv2.findContours(image_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, POLY_APPROX_COEFFICIENT * perimeter, True)
        return perspective_transform(image_p, poly_approx)


def find_lines(image):
    """
    Hough transform is used to detect the lines on the grid
    """
    pass


#sample = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg')
#cv2.imshow("Window", preprocess(cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg')))


processed = preprocess(cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg'))
warped = warp_with_contours(processed)
cv2.imshow("Window", warped)
cv2.waitKey(0)


