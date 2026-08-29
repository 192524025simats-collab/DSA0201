import cv2

# Load video
cap = cv2.VideoCapture("VIDEO.mp4")

if not cap.isOpened():
    print("Error: input.mp4 not found")
    exit()

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# Object classes
classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# Read one video frame
ret, frame = cap.read()

if not ret:
    print("Could not read video frame.")
    cap.release()
    exit()

# Get frame size
h, w = frame.shape[:2]

# Create input blob
blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)),
    0.007843,
    (300, 300),
    127.5
)

# Set input
net.setInput(blob)

# Detect objects
detections = net.forward()

# Vehicle classes
vehicle_classes = [
    "car",
    "bus",
    "motorbike"
]

# Process detections
for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.40:

        class_id = int(detections[0, 0, i, 1])
        label = classes[class_id]

        if label in vehicle_classes:

            box = detections[0, 0, i, 3:7] * [
                w, h, w, h
            ]

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = label + " " + str(
                round(confidence * 100, 1)
            ) + "%"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# Display
cv2.imshow("Vehicle Detection", frame)

# Save output
cv2.imwrite(
    "experiment34_vehicle_detection.jpg",
    frame
)

print("Experiment 34 completed successfully.")

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
