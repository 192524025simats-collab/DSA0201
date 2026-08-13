import cv2

# Read the image
img = cv2.imread("flower.jpg")

# Check image loaded
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Canny Edge Detection", edges)

# Save images
cv2.imwrite("canny_edges.jpg", edges)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
