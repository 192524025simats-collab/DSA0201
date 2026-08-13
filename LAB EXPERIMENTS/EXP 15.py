import cv2

# Read the image
img = cv2.imread("flower.jpg")

# Check image loaded
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator along X-axis
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Apply Sobel operator along Y-axis
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Calculate combined Sobel edges
sobel_xy = cv2.magnitude(sobel_x, sobel_y)

# Convert to unsigned 8-bit image
sobel_xy = cv2.convertScaleAbs(sobel_xy)

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Sobel XY Edge Detection", sobel_xy)

# Save images
cv2.imwrite("sobel_xy_edges.jpg", sobel_xy)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
