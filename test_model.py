import pickle

import cv2

from utils import get_face_landmarks

# Define the list of emotions
emotions = ['HAPPY', 'SAD', 'SURPRISED']

# Load the trained model from file
with open('./model', 'rb') as f:
    model = pickle.load(f)

# Open a video capture object (assuming camera index 0)
cap = cv2.VideoCapture(0)

# Read the first frame from the video capture
ret, frame = cap.read()

# Loop to continuously read frames from the camera
while ret:
    # Read the next frame from the camera
    ret, frame = cap.read()

    # Get face landmarks from the current frame
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    # Check if face landmarks are found
    if face_landmarks:
        # Make a prediction using the trained model
        output = model.predict([face_landmarks])

        # Display the predicted emotion on the frame
        cv2.putText(frame,
                    emotions[int(output[0])],
                    (10, frame.shape[0] - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    5)

    # Display the frame with the emotion label
    cv2.imshow('frame', frame)

    # Wait for a key press and check if it's 'q' to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
