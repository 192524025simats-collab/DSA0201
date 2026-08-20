import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('input.jpg', 0)

# Laplacian mask with positive center coefficient
mask = np.array([[0, -1, 0],
                 [-1, 5, -1],
                 [0, -1, 0]])

# Apply mask
sharpened = cv2.filter2D(img, -1, mask)

# Display images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.show()

# Save output
cv2.imwrite('laplacian_positive.jpg', sharpened)
