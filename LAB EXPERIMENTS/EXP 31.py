import cv2

# Read the original image
img = cv2.imread("watch.jpg")

# Read watch template
template = cv2.imread("watch template.jpg")

if img is None:
    print("Error: input.jpg not found")
    exit()

if template is None:
    print("Error: watch_template.jpg not found")
    exit()

# Convert both images to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Get template size
h, w = template_gray.shape

# Template matching
result = cv2.matchTemplate(
    gray,
    template_gray,
    cv2.TM_CCOEFF_NORMED
)

# Find best matching location
_, max_value, _, max_location = cv2.minMaxLoc(result)

# Matching threshold
threshold = 0.60

if max_value >= threshold:

    # Coordinates
    x, y = max_location

    # Draw rectangle
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Display text
    cv2.putText(
        img,
        "WATCH",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    print("Watch detected.")
    print("Matching score:", max_value)

else:
    print("Watch not detected.")
    print("Matching score:", max_value)

# Display
cv2.imshow("Watch Recognition", img)

# Save
cv2.imwrite("experiment31_watch_detected.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Experiment 31 completed.")
