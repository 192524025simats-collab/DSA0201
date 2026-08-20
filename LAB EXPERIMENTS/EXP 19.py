import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('input.jpg', 0)

# Create blurred image
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Create unsharp mask
mask = cv2.subtract(img, blurred)

# Sharpen the image
sharpened = cv2.add(img, mask)

# Display images
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
plt.title('Sharpened Image')
plt.axis('off')

plt.show()

# Save output
cv2.imwrite('unsharp_masking.jpg', sharpened)
