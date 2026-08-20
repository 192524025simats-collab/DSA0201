import cv2

# Read image
img = cv2.imread('input.jpg')

# Watermark text
text = "WATERMARK"

# Add watermark
cv2.putText(img, text, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 2)

# Display image
cv2.imshow('Watermarked Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('watermarked.jpg', img)
