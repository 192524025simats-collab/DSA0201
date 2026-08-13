import cv2
import numpy as np

# Open the video
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video file
out = cv2.VideoWriter(
    "perspective_transformed_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Source points
    src_points = np.float32([
        [100, 100],
        [w - 100, 100],
        [w - 100, h - 100],
        [100, h - 100]
    ])

    # Destination points
    dst_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # Perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    transformed = cv2.warpPerspective(frame, matrix, (w, h))

    # Display videos
    cv2.imshow("Original Video", frame)
    cv2.imshow("Perspective Transformed Video", transformed)

    # Save transformed frame
  

    # Press q to stop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Perspective transformed video saved successfully!")
