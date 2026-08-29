import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("images.jpg")

if img is None:
    print("Error: input.jpg not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create structuring element
kernel = np.ones((5, 5), np.uint8)

# Black Hat operation
blackhat = cv2.morphologyEx(
    gray,
    cv2.MORPH_BLACKHAT,
    kernel
)

# Display
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blackhat, cmap="gray")
plt.title("Black Hat Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save output
cv2.imwrite("experiment30_blackhat.jpg", blackhat)

print("Experiment 30 completed successfully.")
print("Output saved as experiment30_blackhat.jpg")
