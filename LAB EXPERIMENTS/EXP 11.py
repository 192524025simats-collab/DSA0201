import cv2
import numpy as np

# Read the input image
image = cv2.imread("river.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Define source points
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
src_points = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

# Define destination points
dst_points = np.float32([
    [50, 50],
    [width - 50, 20],
    [width - 20, height - 50],
    [20, height - 20]
])

# Create matrix A for DLT
A = []

for (x, y), (u, v) in zip(src_points, dst_points):
    A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
    A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

A = np.array(A, dtype=np.float64)

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)

# The last row of Vt gives the solution
h = Vt[-1]

# Reshape into a 3x3 Homography Matrix
H = h.reshape(3, 3)

# Normalize the Homography Matrix
H = H / H[2, 2]

print("Homography Matrix obtained using DLT:")
print(H)

# Apply the transformation
transformed_image = cv2.warpPerspective(
    image,
    H,
    (width, height)
)

# Display original image
cv2.imshow("Original Image", image)

# Display transformed image
cv2.imshow("DLT Transformed Image", transformed_image)

# Save transformed image
cv2.imwrite("dlt_transformed.jpg", transformed_image)

# Wait for key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
