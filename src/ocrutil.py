import os

import cv2
import numpy as np
import pytesseract

from imutil import DIM


pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def split_preprocess(image):
    imagec = image.copy()
    gray = cv2.cvtColor(imagec, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return 255 - cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)


def image_to_string2(image):
    digit_str = (pytesseract.image_to_string(image, lang="eng",
                                             config="--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"))
    stripped = ""
    for s in digit_str.split():
        if s.isdecimal():
            stripped = s
            break
    return stripped


def images_to_array(path):
    digits = np.zeros((DIM, DIM), dtype="uint32")

    for i, fname in enumerate(os.listdir(path)):

        invert = split_preprocess(cv2.imread(os.path.join(path, fname)))
        digit = image_to_string2(invert)
        if digit != "":
            digits[int(i / DIM), i % DIM] = int(digit)

    return digits
