import cv2
import matplotlib.pyplot as plt
import numpy as np

#LOADING THE IMAGES TAKEN FROM 3D Slicer
image = cv2.imread('/Users/siddhantagarwal/Desktop/image.jpeg')
mask = cv2.imread('/Users/siddhantagarwal/Desktop/mask.jpeg')

#RESIZING IMAGE TO MAKE SURE IMAGE DIMENSIONS ARE THE SAME
image = cv2.resize(image, (320, 218))
print(image.shape, mask.shape)

#FINDING CONTOURS IN IMAGE
img_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, thresholded = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#ONLY KEEPING THE LARGEST CONTOUR (THE MASK)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

#MAKING A COPY OF THE PROSTATE IMAGE
image_copy = image.copy()

#DRAWING CONTOUR (MASK) ON IMAGE
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

#MASKING THE MASK ONTO THE IMAGE
masked = cv2.bitwise_and(image, mask)

#PRINTING IMAGES
cv2.imwrite("/Users/siddhantagarwal/Desktop/image1.jpg", image)
cv2.imwrite("/Users/siddhantagarwal/Desktop/mask1.jpg", mask)
cv2.imwrite('/Users/siddhantagarwal/Desktop/masked.jpg', masked)
cv2.imwrite('/Users/siddhantagarwal/Desktop/image_copy.jpg', image_copy)