import cv2

img = cv2.imread("image.jpg")

x, y, w, h = cv2.selectROI("Select Object", img, False)

object = img[y:y+h, x:x+w]

cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Selected Object", object)
cv2.imshow("Original Image", img)

cv2.imwrite("extracted_object.jpg", object)

cv2.waitKey(0)
cv2.destroyAllWindows()
