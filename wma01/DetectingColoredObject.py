import cv2
import numpy as np


    
# Define colors range in HSV format
color_ranges = [
    # Red
    ([0, 70, 50], [10, 255, 255], (0, 0, 255)),  
    # Green
    ([36, 70, 50], [70, 255, 255], (0, 255, 0)), 
    # Blue 
    ([110, 70, 50], [130, 255, 255], (255, 0, 0)), 
    # Yellow 
    ([20, 100, 100], [30, 255, 255], (0, 255, 255))  
]


# Load the video
video = cv2.VideoCapture('rgb_ball_720.mp4')

while (video.isOpened() ):
    # Read each frame of the video
    ret, frame = video.read()
    
    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for (lowerBound, upperBound, color) in color_ranges:
        # Create a mask for the color range
        mask = cv2.inRange(hsv, np.array(lowerBound), np.array(upperBound))#The lower and upper variables represent a specific color range in the HSV color space
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#RETR_EXTERNAL  I used RETR_EXTERNAL because I only wanted to extract the outermost contours.
        
        # Draw bounding boxes around the contours
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    
    # Display the resulting frame
    cv2.imshow('color detecting video',frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
