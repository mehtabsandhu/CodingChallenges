import cv2
import numpy as np
import os

# Load the image
image = cv2.imread(os.path.join('perception', 'red.png'))

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

# Reupdates the contours
contours = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Extract all cone points
points = []
for contour in contours:
  for point in contour:
    points.append(point[0])  # Extract (x, y) coordinates

points = np.array(points)

# Separate left and right cones
x_median = np.median(points[:, 0])
left_cones = points[points[:, 0] < x_median]
right_cones = points[points[:, 0] >= x_median]

# Fit lines
left_line = np.polyfit(left_cones[:, 1], left_cones[:, 0], 1)
right_line = np.polyfit(right_cones[:, 1], right_cones[:, 0], 1)

# Define start/end points for lines
y_min = 0
y_max = image.shape[0]
left_x_min, left_x_max = np.polyval(left_line, [y_min, y_max])
right_x_min, right_x_max = np.polyval(right_line, [y_min, y_max])

# Draw the boundary lines
image_lines = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.line(image_lines, (int(left_x_min), int(y_min)), (int(left_x_max), int(y_max)), (255, 0, 0), 3)
cv2.line(image_lines, (int(right_x_min), int(y_min)), (int(right_x_max), int(y_max)), (255, 0, 0), 3)

# Saves the image with the drawn boundary lines 
cv2.imwrite(os.path.join("perception", "answer.png"), cv2.cvtColor(image_lines, cv2.COLOR_RGB2BGR))