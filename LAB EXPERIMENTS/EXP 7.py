import cv2
import numpy as np

# Read the image
image = cv2.imread("flower.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Define three points in the original image
pts1 = np.float32([
    [50, 50],
    [200, 50],
    [50, 200]
])

# Define corresponding points in transformed image
pts2 = np.float32([
    [10, 100],
    [200, 50],
    [100, 250]
])

# Calculate affine transformation matrix
M = cv2.getAffineTransform(pts1, pts2)

# Apply affine transformation
affine_image = cv2.warpAffine(
    image,
    M,
    (width, height)
)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Affine Transformed Image", affine_image)

# Save images

cv2.imwrite("affine_transformed_flower.jpg", affine_image)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
