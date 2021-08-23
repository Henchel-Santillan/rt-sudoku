import os

import cv2
import imutil as im
import ocrutil as ocr
import solver


PATH_TO_RES = r"C:\Users\hench\PycharmProjects\rt-sudoku\res"
DEFAULT_FILENAME = "frame_capture_0.png"


def main():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow("Image Capture")

    if not capture.isOpened():
        capture.open(0)

    while True:
        ret, frame = capture.read()
        img = im.preprocess(frame)
        cv2.imshow('Image Capture', img)

        image = None

        wkey = cv2.waitKey(1)
        if wkey & 0xFF == ord('q'):
            break
        elif wkey & 0xFF == ord('c'):
            image = frame
            cv2.imwrite(os.path.join(PATH_TO_RES, DEFAULT_FILENAME), frame)
            break

    capture.release()
    cv2.destroyAllWindows()

    preprocessed = im.preprocess(image)
    transformed = im.find_contours(image, preprocessed)
    im.partition(transformed)

    board = ocr.image_to_array(os.path.join(PATH_TO_RES, DEFAULT_FILENAME))
    solver.solve(board)
    solver.print_board(board)


if __name__ == "__main__":
    main()
