# open-cv (cv2) learning file number 2
# reference: opencv-python-tutorials (readthedocs.io, py tutorials)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# (1) Accessing and modifying basic image properties (pixel values, shape, size, data type)s
image = cv2.imread('image.jpg')
# modify red value
image.itemset((10, 10, 2), 100)
image.item(10, 10, 2)

print(image.shape)  # Accesses image shape: 3-tuple of (row, column, channel)
print(image.size)   # Returns total number of pixels
print(image.dtype)  # Returns datatype of the image

# Image ROI (Region of Interest) --> indexing

# (2) Splitting and Merging Image Channels
blue, green, red = cv2.split(image)
image = cv2.merge((blue, green, red))

# Alternatively, can index image for split since cv2 natively uses numpy ndarrays (recommended)

# (3) Padding (Border Creation)
image = cv2.imread('image.jpg')
# Border 1: NONE (Original Image)
plt.subplot(231), plt.imshow(image, 'gray'), plt.title('NO BORDER')
# Border 2: Replicate
plt.subplot(232), plt.imshow(image, 'gray'), plt.title("REPLICATE")
# Border 3: Reflect
plt.subplot(233), plt.imshow(image, 'gray'), plt.title("REFLECT")
# Border 4: Reflect_101
plt.subplot(234), plt.imshow(image, 'gray'), plt.title("REFLECT 101")
# Border 5: Wrap
plt.subplot(235), plt.imshow(image, 'gray'), plt.title("WRAP")
# Border 6: Constant
plt.subplot(236), plt.imshow(image, 'gray'), plt.title("CONSTANT")

# (4) Image Blending
# g(x) = (1 - a)f0(x) + af1(x) ==> add() function
# dst = a(Im1) + b(Im2) + y, TYP y = 0 ==> addWeighted()
image1 = cv2.imread('image.jpg')
image2 = cv2.imread('image1.jpg')

dst = cv2.addWeighted(image1, 0.7, image2, 0.3, 0)  # (Im1, a, Im2, b, y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OpenCV add() vs. NumPy add()
num1, num2 = np.uint8([250]), np.uint8([10])
print(cv2.add(num1, num2))   # Returns 255, overflow: 250 + 10 = 260 ==> 255
print(num1 + num2)     # Returns 4, 260 % (255 + 1)

# TYP: Adding images changes colour, blending images changes relative transparency
# TYP: ROI applied to images of rectangular shape

# (5) Superimposing image on another image, bit masking, inverse bit masking
image1 = cv2.imread('image.jpg')
image2 = cv2.imread('image1.jpg')

# Get the shape of the superimposing image
rows, columns, channels = image2.shape
# Set region of interest to top left corner of the superimposed image
roi = image1[0:rows, 0:columns]

# Convert superimposing image to greyscale, apply Binary Threshold
image_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
has_return, m_mask = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)
# Use bitwise_not to get inverse of scaling
mask_inverse = cv2.bitwise_not(m_mask)

# Set scaling of superimposing area (roi) to inverse of mask
image1_background = cv2.bitwise_and(roi, roi, mask=mask_inverse)

# Take only region of superimposing image needed
image2_foreground = cv2.bitwise_and(image2, image2, mask=m_mask)

# Set the left corner to the addition of the superimposing area + region of sp img
dst = cv2.add(image1_background, image2_foreground)
image1[0:rows, 0:columns] = dst

# Show and release
cv2.imshow('Bitwise ImOps', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (6) Thresholding

# (7) Blurring
