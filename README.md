# cv-sudoku
Sudoku solver using rudimentary computer vision and optical character recognition techniques. 
OpenCV learning project.

## Process
An image is first captured via the user's default desktop camera. 
The image is then taken into the script after resources are released for **preprocessing**. 

![Screenshot of sample sudoku puzzle](res/sudoku_grid.jpg)

The **preprocessing** step involves converting the raw image to grayscale, applying a low-pass filter
(in this case, a `Gaussian blur`), and then applying adaptive thresholding to make the image easier to analyze.

![Screenshot of preprocessed sudoku puzzle](res/preprocessed.png)

The next phase attempts to identify the corners of the largest contour in the image and perform a **perspective warp** 
on the image. This is done by generating a kernel of ones (except the matrix corners, which are 0), performing a dilation
morphological operation, finding the contours using the kernel, and then reverse-ordering the contours based on area. Filtering each 
contour based on length and size determines the largest one. The image is then downsampled and a top-down view 
of only the board is obtained.

![Screenshot of top-down view of sudoku puzzle](res/transformed.png)

The downsampled image is then **partitioned**. The warped image is resized to fit a square and is divided into 9 X 9 images, 
each of which represent a cell in the board (some samples below).

![Sample split cell (1)](res/48.png) ![Sample split cell (2)](res/58.png)
![Sample split cell (3)](res/92.png) ![Sample split cell (4)](res/14.png)
![Sample split cell (5)](res/16.png)


Using `PyTesseract`, each of the 81 split images is preprocessed and then analyzed via number whitelisting.
Identified numbers are put into an `ndarray` with shape (9, 9). Once all images have been passed, the numpy
array is converted into a Python list. The `solver` attempts to determine a solution via backtracking, which is 
printed to console.

![Sample sudoku puzzle result](res/final.png)