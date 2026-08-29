import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("flower.jpg")

if img is None:
    print("Error: input.jpg not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create structuring element
kernel = np.ones((5, 5), np.uint8)

# Top Hat operation
tophat = cv2.morphologyEx(
    gray,
    cv2.MORPH_TOPHAT,
    kernel
)

# Display
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(tophat, cmap="gray")
plt.title("Top Hat Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save output
cv2.imwrite("experiment29_tophat.jpg", tophat)

print("Experiment 29 completed successfully.")
print("Output saved as experiment29_tophat.jpg")
