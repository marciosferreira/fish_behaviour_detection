# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('C:/Users/marcio/Videos/Ian_videos/20191121_1454_iCab_L_C.avi')

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

frame_n = 1
# Read until video is completed
while(cap.isOpened()):
    
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Frame', frame)
        print(frame_n)
        print("shape")
        print(frame.shape)
        frame_n +=1
  
        
        
        
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()