import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('input.jpg', 0)

# Sobel gradient masks
Gx = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])

Gy = np.array([[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]])

# Apply gradient masks
gradient_x = cv2.filter2D(img, cv2.CV_64F, Gx)
gradient_y = cv2.filter2D(img, cv2.CV_64F, Gy)

# Calculate gradient magnitude
gradient = cv2.magnitude(gradient_x, gradient_y)

# Convert gradient to uint8
gradient = cv2.normalize(gradient, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)

# Sharpen image
sharpened = cv2.add(img, gradient)

# Display images
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gradient, cmap='gray')
plt.title('Gradient')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.show()

# Save output
cv2.imwrite('gradient_sharpened.jpg', sharpened)
