import cv2

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    capture.open(0)

while True:
    ret, frame = capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image Capture', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()