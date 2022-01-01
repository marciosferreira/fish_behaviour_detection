import cv2
import numpy as np

# Load image, create mask, and draw white circle on mask


# Load the images
img1 = cv2.imread('C:/Users/marci/Pictures/Camera Roll/WIN_20210225_13_09_26_Pro.jpg')
img2 = cv2.imread('C:/Users/marci/Pictures/Camera Roll/WIN_20210225_13_09_26_Pro.jpg')

# Convert it to HSV
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# find the metric value
metric_val = cv2.compareHist(hist_img2, hist_img2, cv2.HISTCMP_INTERSECT)
print(metric_val)
