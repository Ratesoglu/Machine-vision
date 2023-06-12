import cv2
import numpy as np

# Harris corner detection parameters
block_size = 2
ksize = 5
k = 0.04
threshold = 0.01

# Load video
cap = cv2.VideoCapture('cleaning_bot.mp4')

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Stop if video is finished
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Harris corner response
    dst = cv2.cornerHarris(gray, block_size, ksize, k)

    # Threshold corner response
    dst[dst < threshold * dst.max()] = 0

    # Dilate corners to make them more visible
    dst = cv2.dilate(dst, None)

    # Draw corners as red dots
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]
    frame = cv2.dilate(frame, None, iterations=1) # Thickening the lines

    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Draw edges as white lines
    frame[edges > 0] = [255, 255, 255]
    frame = cv2.dilate(frame, None, iterations=1) # Thickening the lines
    
      # resize the window to 960 x 540 pixels
    frame = cv2.resize(frame, (1280, 720))

    # Display result
    cv2.imshow('White lines and Red dots', frame)

    # Exit if user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
