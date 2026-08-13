import cv2
import numpy as np

# Read the input image
image = cv2.imread("mickey.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Define four corresponding points in the original image
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
src_points = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

# Define the corresponding destination points
dst_points = np.float32([
    [50, 50],
    [width - 50, 20],
    [width - 20, height - 50],
    [20, height - 20]
])

# Calculate the Homography Matrix
H, status = cv2.findHomography(src_points, dst_points)

# Apply the Homography transformation
transformed_image = cv2.warpPerspective(
    image,
    H,
    (width, height)
)

# Display the original image
cv2.imshow("Original Image", image)

# Display the transformed image
cv2.imshow("Homography Transformed Image", transformed_image)

# Save the transformed image
cv2.imwrite("homography_transformed.jpg", transformed_image)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
