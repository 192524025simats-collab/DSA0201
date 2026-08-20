import cv2

# Read images
img1 = cv2.imread('input.jpg')
img2 = cv2.imread('destination.jpg')

# Crop region
crop = img1[50:200, 50:200]

# Paste crop into destination image
img2[50:200, 50:200] = crop

# Display result
cv2.imshow('Cropped and Pasted Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('copy_paste.jpg', img2)
