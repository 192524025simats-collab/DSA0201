import cv2

# Read the image
image = cv2.imread("house.jpg")

# Check if the image is loaded successfully
if image is None:
    print("Error: Unable to read the image.")
else:
    # Apply Gaussian Blur
    # (kernel size should be positive and odd, e.g., (5,5), (9,9), (15,15))
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # Display the original and blurred images
    cv2.imshow("Original Image", image)
    cv2.imshow("Blurred Image", blurred_image)

    # Save the blurred image
    cv2.imwrite("blurred_image.jpg", blurred_image)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
