"""
Title: How can we use Hough transform to detect road lines?
Implementation in Python using OpenCV library

Author: Hemant Ramphul
Date: 28 September 2022

Université des Mascareignes (UdM)
Faculty of Information and Communication Technology
Master Artificial Intelligence and Robotics

Official Website: https://udm.ac.mu
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

path = 'images/original2.jpg'  # Absolute path of the image
original = mpimg.imread(path)  # Read image from path
color = (0, 255, 0)  # Line color

# Gaussian smoothing: remove noise using low pass filter
filter_size = 13

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Hough Transform
rho = 1  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 155  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5  # minimum number of pixels making up a line
max_line_gap = 250  # maximum gap in pixels between connectable line segments


def grayscale(image):
    """
    Applies the Grayscale transform

    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call [plt.imshow(image, cmap='gray')]

    :param image: Image, camera frame
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny(image, low_threshold_gradient, high_threshold_gradient):
    """
    Applies the Canny transform

    :param image: Image, camera frame
    :param low_threshold: Low threshold value of intensity gradient.
    :param high_threshold: High threshold value of intensity gradient.
    :return: detection of edges image
    """
    return cv2.Canny(image, low_threshold_gradient, high_threshold_gradient)


def gaussian_blur(image, kernel_size):
    """
    Applies a Gaussian Noise kernel

    :param image: Image, camera frame
    :param kernel_size: filter size
    :return: blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def hough_lines_p(image, rho, theta, threshold, min_line_length, max_line_gap):
    """
    Creates hough line
    Note that: `image` should be the output of a Canny transform.

    :param image: Image, camera frame
    :return: hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)


# Draw line on image function with four parameters
# img:[image], lines:[lines], color:[default_value=[0, 255, 0]], thickness:[default_value=2]
def draw_lines(image, lines, color=[0, 255, 0], thickness=2):
    """
    Draw line on image function with four parameters

    :param image: Image, camera frame
    :param lines: detect lines
    :param color: color of lines [default_value=[0, 255, 0]]
    :param thickness: thickness of lines [default_value=2]
    :return: an image with hough lines drawn.
    """
    # In case of error, don't draw the line(s)
    if lines is None:
        return
    if len(lines) == 0:
        return
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw lines on the image
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def weighted_img(image, initial_img, α=0.8, β=1., λ=0.):
    """
        Combine the two images
    """
    return cv2.addWeighted(initial_img, α, image, β, λ)


def draw_lines_p(image, lines, color=[0, 255, 0], thickness=2):
    """
        Draw line on image function with four parameters

        :param image: Image, camera frame
        :param lines: detect lines
        :param color: color of lines [default_value=[0, 255, 0]]
        :param thickness: thickness of lines [default_value=2]
        :return: an image with hough lines drawn.
    """
    # In case of error, don't draw the line(s)
    if lines is None:
        return
    if len(lines) == 0:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 > 400 or y2 > 400:  # Filter out the lines in the top of the image
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


# Read image from path
image = mpimg.imread(path)

# Convert the image to grayscale
gray_image = grayscale(image)
# Apply Gaussian smoothing
blurred_image = gaussian_blur(gray_image, filter_size)
# Apply Canny Edge Detector
canny_image = canny(blurred_image, low_threshold, high_threshold)

# Hough Lines
hough_lines = cv2.HoughLines(canny_image, rho, theta, threshold)
# Return an array of zeros with the same shape and type as a given array.
hough_lines_image = np.zeros_like(image)
# Draw lane lines on the original image
draw_lines(hough_lines_image, hough_lines)
# Combine the two images
original_image_with_hough_lines = weighted_img(hough_lines_image, image)

# Hough Lines P
# Detect points that form a line
# Run Hough on edge detected image
lines = hough_lines_p(canny_image, rho, theta, threshold, min_line_length, max_line_gap)
# Draw detected line into image
draw_lines_p(image, lines, color=[255, 255, 0])

# Use matplotlib.pyplot to print our result with at different stages of process
plt.figure('Detect Road Line with Hough Transform: Hemant Ramphul', figsize=(50, 50))
plt.subplot(221)
plt.axis('off')
plt.title("Original Image")
plt.imshow(original)
plt.subplot(222)
plt.axis('off')
plt.title("Edges Image")
plt.imshow(canny_image, cmap='gray')
plt.subplot(223)
plt.axis('off')
plt.title("Image with Hough Lines")
plt.imshow(original_image_with_hough_lines)
plt.subplot(224)
plt.axis('off')
plt.title("Image with hough lines using HoughLinesP")
plt.imshow(image)
plt.show()
