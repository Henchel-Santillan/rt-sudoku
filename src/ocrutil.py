import numpy as np
import pytesseract
from PIL import Image
from imutil import DIM


def image_to_array(pil_images):
    """
    Reads in an array of PIL images, recognizes the digits in each image, and writes to a returned 2D list.
    """
    arr = np.zeros((DIM, DIM), np.uint32)
    for image in pil_images:
        digit_str = (pytesseract.image_to_string(image, lang="eng",
                                                 config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"))

