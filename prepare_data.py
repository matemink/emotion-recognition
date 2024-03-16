import os

import cv2
import numpy as np

from utils import get_face_landmarks  # Assuming this is a custom utility function to get face landmarks

# Directory containing the image data
data_dir = './content/faces/'

output = []
# Loop through each emotion directory
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    # Loop through each image in the emotion directory
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        # Read the image
        image = cv2.imread(image_path)

        # Get the face landmarks for the image
        face_landmarks = get_face_landmarks(image)

        # Check if the correct number of landmarks were detected (1404 in this case)
        if len(face_landmarks) == 1404:
            # Append the emotion index to the landmarks list
            face_landmarks.append(int(emotion_indx))
            # Add the landmarks to the output list
            output.append(face_landmarks)

# Save the output as a numpy array to a text file
np.savetxt('data.txt', np.asarray(output))
