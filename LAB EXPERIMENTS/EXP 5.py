import cv2

# Read the image
image = cv2.imread("mickey.jpg")

# Check if image is loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Rotate image 90 degrees clockwise
clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Rotate image 90 degrees counter-clockwise
counter_clockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Clockwise Rotation", clockwise)
cv2.imshow("Counter-Clockwise Rotation", counter_clockwise)

# Save images
cv2.imwrite("clockwise_rotation.jpg", clockwise)
cv2.imwrite("counter_clockwise_rotation.jpg", counter_clockwise)

print("Images saved successfully!")

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
