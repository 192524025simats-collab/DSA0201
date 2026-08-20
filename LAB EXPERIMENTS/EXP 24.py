import cv2
import numpy as np

# Read image
img = cv2.imread('input.jpg', 0)

# Structuring element
kernel = np.ones((5, 5), np.uint8)

# Erosion
eroded = cv2.erode(img, kernel, iterations=1)

# Display
cv2.imshow('Erosion', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('erosion.jpg', eroded)
