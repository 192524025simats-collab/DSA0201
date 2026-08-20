import cv2
import numpy as np

# Read image
img = cv2.imread('input.jpg', 0)

# Convolution kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply convolution
boundary = cv2.filter2D(img, -1, kernel)

# Display
cv2.imshow('Boundary', boundary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('boundary.jpg', boundary)
