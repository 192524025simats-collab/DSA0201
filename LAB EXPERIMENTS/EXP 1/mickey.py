import cv2
import numpy as np

# Read the image
image = cv2.imread("mickey.jpg")

# Check if the image is loaded successfully
if image is None:
    print("Error: Unable to read the image.")
else:
    # Create a kernel (structuring element)
    kernel = np.ones((5, 5), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # Display the original and eroded images
    cv2.imshow("Original Image", image)
    cv2.imshow("Eroded Image", eroded_image)

    # Save the eroded image
    cv2.imwrite("eroded_image.jpg", eroded_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
