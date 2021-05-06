# open-cv (cv2) learning file number 1
# reference: opencv-python-tutorials (readthedocs.io, py tutorials)

# TODO: Create directory for tutorial resources (i.e images, videos) --> res

import cv2
import numpy as np
from matplotlib import pyplot as plt

# (1): read in image, write image, save image

image = cv2.imread('image.jpg', 0)  # opens image in greyscale (0 arg)
cv2.imshow('Image Show', image)     # displays image in a window
key = cv2.waitKey(0) & 0xFF         # waits indefinitely for key stroke, 64-bit machine

if key == 27:                       # ESC keycode
    cv2.destroyAllWindows()         # destroys created window(s)

elif key == ord('s'):               # 's' key for save and exit bind
    cv2.imwrite('image_new.png', image)
    cv2.destroyAllWindows()

# (2) display image using matplotlib
# NOTE: open-cv is BGR, matplotlib is RGB

image = cv2.imread("image.jpg")
blue, green, red = cv2.split(image)         # split image into bgr
image2 = cv2.merge([red, green, blue])      # create new image merged in rgb order

plt.subplot(121)            # (nrows, ncols, index)
plt.imshow(image)           # distorted colour
plt.subplot(122)
plt.imshow(image2)          # true colour
plt.show()                  # show the plots

cv2.imshow('BGR Image', image)      # true colour
cv2.imshow('RGB Image', image2)     # distorted colour
cv2.waitKey(0)
cv2.destroyAllWindows()

# (3)   Capture video from camera, convert to greyscale, and display
capture = cv2.VideoCapture(0)       # init VideoCapture obj; number specifies which camera

if not capture.isOpened():
    capture.open()

while True:
    has_return, frame = capture.read()  # flag for return, with frame returned
    capture_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts frame to greyscale
    cv2.imshow('Frame', capture_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):       # exit if 'q' pressed
        break

capture.release()                       # release camera resource
cv2.destroyAllWindows()

# (4) Playing Video from file

capture = cv2.VideoCapture('video.avi')     # .avi recommended; str passed as arg for local path to file

while capture.isOpened():
    has_return, frame = capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Frame', image_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# (5) Saving video
capture = cv2.VideoCapture(0)
# four character code of codec used to compress frames
code = cv2.VideoWriter_fourcc(*'XVID')
# (file name, fourcc, fps, 2-tuple frame size)
output = cv2.videoWriter('output.avi', code, 20.0, (640, 480))

while capture.isOpened():
    has_return, frame = capture.read()

    # get frame if returned, flip to vertical direction, and save
    if has_return:
        frame = cv2.flip(frame, 0)

    output.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # break from loop if no return
    else:
        break

# release and destroy all resources
capture.release()
output.release()
cv2.destroyAllWindows()
