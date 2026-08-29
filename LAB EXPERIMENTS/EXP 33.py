import cv2

# Read image
img = cv2.imread("images.jpg")

if img is None:
    print("Error: input.jpg not found")
    exit()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw rectangles around faces
for (x, y, w, h) in faces:

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

print("Number of faces detected:", len(faces))

# Display
cv2.imshow("Face Detection", img)

# Save
cv2.imwrite("experiment33_face_detection.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Experiment 33 completed successfully.")
