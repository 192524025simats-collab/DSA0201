
import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 1 for Normal Speed")
print("Press 2 for Slow Motion")
print("Press 3 for Fast Motion")
print("Press q to Quit")

# Default speed
delay = 30

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read frame")
        break

    # Display the captured video
    cv2.imshow("Webcam Video", frame)

    # Get keyboard input
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('1'):
        # Normal speed
        delay = 30
        print("Normal Speed")

    elif key == ord('2'):
        # Slow motion
        delay = 100
        print("Slow Motion")

    elif key == ord('3'):
        # Fast motion
        delay = 1
        print("Fast Motion")

    elif key == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
