import cv2

# Read the image
image = cv2.imread("river.jpg")

# Check if the image is loaded successfully
if image is None:
    print("Error: Unable to read the image.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Display the original and edge-detected images
    cv2.imshow("Original Image", image)
    cv2.imshow("Canny Edge Detection", edges)

    # Save the output image
    cv2.imwrite("outline_image.jpg", edges)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
