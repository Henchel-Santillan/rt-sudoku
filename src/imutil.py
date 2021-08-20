import cv2
import numpy as np
import operator

KERNEL_SIZE = (5, 5)
POLY_APPROX_COEFFICIENT = 0.015
ARB_RHO_THRESH = 15
ARB_THETA_THRESH = 0.1
DIM = 9


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

    return perspective_transform(src, max_c)


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


def detect_cells(image_t):
    canny = cv2.Canny(cv2.cvtColor(image_t, cv2.COLOR_BGR2GRAY), 50, 110)
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

    kernel_h = np.ones((1, 20), np.uint8)
    morph_h = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_h)

    kernel_v = np.ones((20, 1), np.uint8)
    morph_v = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_v)

    morph = morph_h | morph_v
    return cv2.dilate(morph, np.ones((3, 3), np.uint8), iterations=1)


def extract_cells(image_t):
    thresh = preprocess(image_t)
    cs_thresh, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cs_thresh = cs_thresh[0] if len(cs_thresh) == 2 else cs_thresh[1]

    for c in cs_thresh:
        if cv2.contourArea(c) < 1000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    cv2.imshow("", thresh)
    cv2.waitKey(0)


def snip_rect(image, rect):
    return image[int(rect[0][1]): int(rect[1][1]), int(rect[0][0]): int(rect[1][0])]


def scale_and_centre(image, size, margin=0, background=0):
    height, width = image.shape[:2]

    def pad_centre(length):
        s1 = int((size - length) / 2)
        return (s1, s1) if length % 2 == 0 else (s1, s1 + 1)

    if height > width:
        top = int(margin / 2)
        bottom = top
        r = (size - margin) / height
        width, height = r * width, r * height
        left, right = pad_centre(width)
    else:
        left = int(margin / 2)
        right = left
        r = (size - margin) / width
        width, height = r * width, r * height
        top, bottom = pad_centre(height)

    image = cv2.resize(image, (width, height))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(image, (size, size))


def deduce_grid(image):
    squares = []
    side = image.shape[:1][0] / DIM

    for y in range(DIM):
        for x in range(DIM):
            p1, p2 = (x * side, y * side), ((x + 1) * side, (y + 1) * side)
            squares.append((p1, p2))
    return squares


def find_largest_blob(image, top_left=None, bottom_right=None):
    copy_image = image.copy()
    height, width = copy_image.shape[:2]

    max_area = 0
    seed = (None, None)

    if top_left is None:
        top_left = [0, 0]
        
    if bottom_right is None:
        bottom_right = [width, height]

    for x in range(top_left[0], bottom_right[0]):
        for y in range(top_left[1], bottom_right[1]):
            if copy_image.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(copy_image, None, (x, y), 64)[0]
                if area > max_area:
                    max_area = area
                    seed = (x, y)

    for x in range(width):
        for y in range(height):
            if copy_image.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(copy_image, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)

    if all([c is not None for c in seed]):
        cv2.floodFill(copy_image, mask, seed, 255)

    top, bottom = height, 0
    left, right = width, 0

    for x in range(width):
        for y in range(height):
            if copy_image.item(y, x) == 64:
                cv2.floodFill(image, mask, (x, y), 0)

            if copy_image.item(y, x) == 255:
                if y < top: top = y
                if y > bottom: bottom = y
                if x < left: left = x
                if x > right: right = x

    return image, np.array([[left, top], [right, bottom]], dtype=np.float32), seed


def extract_digit(image, rect, size):
    digit = snip_rect(image, rect)
    height, width = digit.shape[:2]
    margin = int(np.mean([height, width]) / 2.5)

    _, bounding, seed = find_largest_blob(image, [margin, margin], [width - margin, height - margin])
    digit = snip_rect(image, bounding)

    width = bounding[1][0] - bounding[0][0]
    height = bounding[1][1] - bounding[0][1]

    if width > 0 and height > 0 and (width * height) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    return np.zeros((size, size), np.uint8)


def find_digits(image, squares, size):
    digits = []
    image = preprocess(image)
    for s in squares:
        digits.append(extract_digit(image, s, size))
    return digits


def fix_image(digits, color=255):
    rows = []
    border = [cv2.copyMakeBorder(image.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, color) for image in digits]
    for x in range(DIM):
        rows.append(np.concatenate(border[x * DIM: ((x + 1) * DIM)], axis=1))
    return np.concatenate(rows)


sample = cv2.imread(r'C:\Users\hench\PycharmProjects\rt-sudoku\res\sudoku_grid.jpg')
transformed = find_contours(sample, preprocess(sample))
digits_f = find_digits(transformed, deduce_grid(transformed), 28)
resized = scale_and_centre(transformed, 28)

#fixed = fix_image(digits_f)
cv2.imshow("Fixed Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

