# 16. Perform Sharpening of Image using Laplacian mask with negative center coefficient

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read input image
img = cv2.imread('input.jpg', 0)

# Define Laplacian mask with negative center coefficient
laplacian_mask = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

# Apply Laplacian filter
laplacian = cv2.filter2D(img, -1, laplacian_mask)

# Sharpen the image
sharpened = img - laplacian

# Display results
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.show()

# Save output image
cv2.imwrite('sharpened_image.jpg', sharpened)
