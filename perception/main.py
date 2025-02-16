import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread(os.path.join('perception', 'red.png'))

print(image.shape)

mid = image.shape[1] // 2

left_image = image[:, :mid]
right_image = image[:, mid:]

print(left_image.shape)

"""

Split image into half down the middle

"""

# Convert to RGB (since OpenCV loads in BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV for better color segmentation
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for red cones (adjust if necessary)
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Create masks to detect red color in the image
mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Filter contours by area
min_contour_size = 150
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_size]

# Create a new mask with only large contours
filtered_mask = np.zeros_like(mask_cleaned)
cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Draw contours on the original image
image_contours = image_rgb.copy()
cv2.drawContours(image_contours, filtered_contours, -1, (0, 255, 0), 2)

# Display the mask and detected contours
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(filtered_mask, cmap="gray")
axes[0].set_title("Binary Mask of Red Cones")
axes[0].axis("off")

axes[1].imshow(image_contours)
axes[1].set_title("Detected Contours")
axes[1].axis("off")

plt.show()