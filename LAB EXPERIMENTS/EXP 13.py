import cv2

# Read the image
img = cv2.imread("mickey.jpg")

# Check image loaded
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator along X-axis
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Convert to unsigned 8-bit image
sobel_x = cv2.convertScaleAbs(sobel_x)

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Sobel X Edge Detection", sobel_x)

# Save images
cv2.imwrite("original_mickey.jpg", img)
cv2.imwrite("sobel_x_edges.jpg", sobel_x)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
