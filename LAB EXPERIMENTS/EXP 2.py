import cv2

video = "video.mp4"

def save_video(speed, output_name):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video
    out = cv2.VideoWriter(
        output_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)

        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(speed) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(output_name, "saved successfully!")


print("1. Normal Speed")
save_video(30, "normal_video.mp4")

print("2. Slow Motion")
save_video(100, "slow_motion_video.mp4")

print("3. Fast Motion")
save_video(5, "fast_motion_video.mp4")
