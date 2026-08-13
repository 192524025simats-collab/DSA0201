import cv2

# Read the image
img = cv2.imread("house.jpg")

# Check image loaded
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator along Y-axis
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert to unsigned 8-bit image
sobel_y = cv2.convertScaleAbs(sobel_y)

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Sobel Y Edge Detection", sobel_y)

# Save images

cv2.imwrite("sobel_y_edges.jpg", sobel_y)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
