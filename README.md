## Lane Detection Project


The goal of this project is to create a pipeline for detecting lanes in a video. 

The pipeline includes:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw video frames.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify the binary image to bird eye view.
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center and approximate steering angle.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The result is shown in https://www.youtube.com/watch?v=SzF6Eo9G4U0
