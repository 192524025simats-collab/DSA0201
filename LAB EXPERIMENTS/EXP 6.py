import cv2
import numpy as np

# Read the image
image = cv2.imread("mickey.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Translation values
tx = 100   # Move right
ty = 50    # Move down

# Create translation matrix
M = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

# Apply translation
moved_image = cv2.warpAffine(image, M, (width, height))

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Moved Image", moved_image)

# Save images

cv2.imwrite("translated_mickey.jpg", moved_image)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
