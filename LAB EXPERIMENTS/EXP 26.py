import cv2
import numpy as np

# Read image
img = cv2.imread('input.jpg', 0)

# Structuring element
kernel = np.ones((5, 5), np.uint8)

# Opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Display
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('opening.jpg', opening)
