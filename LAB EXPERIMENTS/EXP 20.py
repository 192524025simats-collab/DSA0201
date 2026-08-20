import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('input.jpg', 0)

# Blur the image
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# High-frequency mask
mask = cv2.subtract(img, blurred)

# High-boost factor (A >= 1)
A = 2

# High-boost sharpening
sharpened = cv2.addWeighted(img, A, blurred, -(A - 1), 0)

# Display images
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
plt.title('High-Boost Sharpened')
plt.axis('off')

plt.show()

# Save output
cv2.imwrite('high_boost_sharpened.jpg', sharpened)
