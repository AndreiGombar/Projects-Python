import cv2
import numpy as np
from matplotlib import pyplot as plt

# ------------------ Hough Line Transform ------------------ #
def Hough_Transform(img):

    # image dimensions
    height, width = img.shape[:2]

    # maximum value of rho
    rho_max = int(round(np.sqrt(np.square(height) + np.square(width))))

    # rho values
    rhos = np.arange(-rho_max/2, rho_max)

    # theta values range from -pi/2 to pi
    thetas = np.deg2rad(np.arange(-90, 180))

    M = rho_max
    N = len(thetas)

    Hough_array = np.zeros((M, N))

    max_bins = 0

    for x in range(width):
        for y in range(height):
           if all(item == 255 for item in img[y,x]):

               for n in range(N):
                phi = np.arctan(y/x)
                thetas[n] = 180 - ((2*180*(N-n))/(N-1))

                if (thetas[n] >= phi - 180/2) and (thetas[n] <= phi + 180/2):
                   # Rho values can be calculated with the formula: rho= x*cos(theta)+y*sin(theta)
                   rho = x*np.cos(thetas[n]) + y*np.sin(thetas[n])

                   # m index of the quantized rho value
                   m = round(M - ((rho_max - rho) * (M - 1) / (rho_max)))

                   # Increment the Hough array
                   Hough_array[m,n] += 1

                   if Hough_array[m,n] > max_bins:
                        max_bins = Hough_array[m,n]


    return Hough_array, rho_max, rhos, thetas


# ------------------ Line Detection ------------------ #
def Detect_HoughLines(img, Hough_array, rho_max, thetas):

    # Minimum number of intersections
    threshold = 100

    for rho in range(Hough_array.shape[0]):
        for theta in range(Hough_array.shape[1]):

            # Detect lines (ρ, θ) pairs that have a number of intersections larger than a threshold
            if Hough_array[rho, theta] >= threshold:
                th = thetas[theta]
                r = rho - rho_max

                a = np.cos(th)
                b = np.sin(th)

                x0 = a * r
                y0 = b * r

                # Represent the coordinates of line detected
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Draw a red line on the input image
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img


# ------------------ Display ------------------ #
def display(img, hough_line, hough_transform):

    houghline_negative, houghtransform_negative, img_negative = negativationImage(img, hough_line, hough_transform)

    # show result
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(img_negative, cmap='inferno'), plt.title("Original Image")


    plt.subplot(132)
    plt.imshow(houghline_negative, cmap='inferno'), plt.title("Hough Lines Detected")

    plt.subplot(133)
    plt.title("Hough Transform")
    plt.xlabel('Theta'),    plt.imshow(houghtransform_negative, cmap='inferno')

    plt.ylabel('Rho')

    plt.show()


# ------------------ Negativation Image ------------------ #
def negativationImage(img, hough_line, hough_transform):

    # For better visualization we used the negativation of image
    img_shape = np.shape(hough_transform)
    img_negative = np.zeros(img_shape)
    img_negative = 255 - img

    houghline_shape = np.shape(hough_line)
    houghline_negative = np.zeros(houghline_shape)
    houghline_negative = 255 - hough_line

    houghtransform_shape = np.shape(hough_transform)
    houghtransform_negative = np.zeros(houghtransform_shape)
    houghtransform_negative = 255 - hough_transform

    return houghline_negative, houghtransform_negative, img_negative


# -------- main -------- #
if __name__ == '__main__':

    # read image 1
    img1_path = "Image_1.png"

    img1 = cv2.imread(img1_path)
    img1_copy = cv2.imread(img1_path)

    # Compute Hough Transform & Line detection of image 1
    hough_transform1, rho_max1, rho1, theta1 = Hough_Transform(img1_copy)
    hough_line1 = Detect_HoughLines(img1_copy, hough_transform1, rho_max1, theta1)


    # read image 2
    img2_path = "Image_2.png"

    img2 = cv2.imread(img2_path)
    img2_copy = cv2.imread(img2_path)

    # Compute Hough Transform & Line detection of image 2
    hough_transform2, rho_max2, rho2, theta2 = Hough_Transform(img2_copy)
    hough_line2 = Detect_HoughLines(img2_copy, hough_transform2, rho_max2, theta2)

    # Display original image, Lines detected, Hough Transform
    display(img1, hough_line1, hough_transform1)
    display(img2, hough_line2, hough_transform2)
