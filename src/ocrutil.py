import os

import pytesseract
from imutil import DIM


def image_to_array(path):
    digits = [0] * (DIM ** 2)
    i = 0

    for image in os.listdir(path):
        fname = os.fsdecode(image)
        if fname.endswith(".png"):
            digit_str = (pytesseract.image_to_string(image, lang="eng",
                                                     config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"))
            if digit_str != "" and digit_str.isdecimal():
                digits[i] = int(digit_str)

    return digits
