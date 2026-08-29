import cv2

# Open video
cap = cv2.VideoCapture("VIDEO.mp4")

if not cap.isOpened():
    print("Error: input.mp4 not found")
    exit()

# Store all frames
frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frames.append(frame)

cap.release()

print("Total frames:", len(frames))

# Get video properties
if len(frames) == 0:
    print("No frames found.")
    exit()

height, width = frames[0].shape[:2]

# Get original FPS
cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if fps <= 0:
    fps = 30

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "experiment32_reverse.mp4",
    fourcc,
    fps,
    (width, height)
)

# Play frames in reverse
for frame in reversed(frames):

    cv2.imshow("Reverse Video", frame)

    # Save frame
    out.write(frame)

    # Press Q to stop
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

out.release()
cv2.destroyAllWindows()

print("Experiment 32 completed successfully.")
print("Reverse video saved as experiment32_reverse.mp4")
