import cv2
import imutil as im
import solver

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    capture.open(0)

while True:
    ret, frame = capture.read()
    img = im.preprocess(frame)
    cv2.imshow('Image Capture', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
capture.release()
cv2.destroyAllWindows()

