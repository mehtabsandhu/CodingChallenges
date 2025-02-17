# Importing the modules
import cv2
import numpy as np
import os

# Loads the image
image = cv2.imread(os.path.join('perception', 'red.png'))

# Converts the image to HSV for better color segmentation
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Defining the color ranges for the color red.
# In the HSV format, red exists between the H ranges of 0-10 and 160-180
# I've tweaked the HSV ranges to include as many cone pixels while leaving out reds from the surroundings (namely the lights)
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Creates masks to detect red color in the image
mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)

# Combines the two masks together
# Accomplishes this by checking if either bit is 1 (similar to an "OR" statement)
mask = cv2.bitwise_or(mask1, mask2)

# Applies morphological operations to clean up the mask
# It first removes small noise (extremely small pixels of red randomly scattered across the image)
# It then closes any void gaps between large group of red pixels (filling them in with red as well)
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

# Locates the boundaries of the red objects in the image
contours = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Filters contours to only keep red objects with an area of more than 150 pixels
min_contour_size = 150
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_size]

# Creates a new mask with only large contours
filtered_mask = np.zeros_like(mask_cleaned)
cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Draws contours on the original image
image_contours = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(image_contours, filtered_contours, -1, (0, 255, 0), 2)

# Reupdates the contours
contours = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Extracts all cone points
points = []
for contour in contours:
  for point in contour:
    points.append(point[0])  # Extract (x, y) coordinates

points = np.array(points)

# Seperates the image into two vertical halves, to calculate the two lines individually
center_coord = np.median(points[:, 0])
left_cones = points[points[:, 0] < center_coord]
right_cones = points[points[:, 0] >= center_coord]

# Calculates the best fit line for both vertical halves seperately
left_line = np.polyfit(left_cones[:, 1], left_cones[:, 0], 1)
right_line = np.polyfit(right_cones[:, 1], right_cones[:, 0], 1)

# Defines start/end points for lines
y_min = 0
y_max = image.shape[0]
left_x_min, left_x_max = np.polyval(left_line, [y_min, y_max])
right_x_min, right_x_max = np.polyval(right_line, [y_min, y_max])

# Draws the boundary lines on a new image
image_lines = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.line(image_lines, (int(left_x_min), int(y_min)), (int(left_x_max), int(y_max)), (255, 0, 0), 3)
cv2.line(image_lines, (int(right_x_min), int(y_min)), (int(right_x_max), int(y_max)), (255, 0, 0), 3)

# Saves the image with the drawn boundary lines to "answer.png"
cv2.imwrite(os.path.join("perception", "answer.png"), cv2.cvtColor(image_lines, cv2.COLOR_RGB2BGR))