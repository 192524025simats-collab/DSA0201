import cv2
import numpy as np

# Read the input image
image = cv2.imread("house.jpg")

# Check if image is loaded
if image is None:
    print("Error: Image not found!")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Define source points
src_points = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

# Define destination points
dst_points = np.float32([
    [100, 50],
    [width - 100, 0],
    [width - 50, height - 1],
    [50, height - 50]
])

# Calculate perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply perspective transformation
transformed_image = cv2.warpPerspective(
    image,
    matrix,
    (width, height)
)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Perspective Transformed Image", transformed_image)

# Save images

cv2.imwrite("perspective_transformed_house.jpg", transformed_image)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
