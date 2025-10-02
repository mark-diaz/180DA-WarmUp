'''
This python script uses open cv to
1. Track black objects in a given frame
2. Get the dominant color in a central rectangle in the center of the frame

'''

import cv2
import numpy as np

def get_dominant_color(image, k=3):
    # Return dominant color in BGR format using K-Means clustering
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = centers[np.argmax(counts)]
    return np.uint8(dominant)

# Video capture
cap = cv2.VideoCapture(0)

# HSV range for black object
lower_hsv = np.array([0, 0, 0])
upper_hsv = np.array([180, 255, 50])
min_area = 2000  # Minimum contour area

# Central rectangle for dominant color
rect_size = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    x1, y1 = cx - rect_size // 2, cy - rect_size // 2
    x2, y2 = cx + rect_size // 2, cy + rect_size // 2

    # -------------------------------
    # Part 1: Black object tracking
    # -------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if large_contours:
            c = max(large_contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    # -------------------------------
    # Part 2: Dominant color in center
    # -------------------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]
    dominant_color = get_dominant_color(roi, k=3)

    # Display dominant color as rectangle and text
    cv2.rectangle(frame, (10, 10), (60, 60), tuple(int(c) for c in dominant_color), -1)
    color_text = f'Dominant BGR: {dominant_color}'
    cv2.putText(frame, color_text, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # Show results
    cv2.imshow('Frame', frame)
    # cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
