import cv2
import mediapipe as mp
from pathlib import Path
import pandas as pd

# Define the signing alphabet
signing_alphabet = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'k': 9,
    'l': 10,
    'm': 11,
    'n': 12,
    'o': 13,
    'p': 14,
    'q': 15,
    'r': 16,
    's': 17,
    't': 18,
    'u': 19,
    'v': 20,
    'w': 21,
    'x': 22,
    'y': 23
}

# Define the directory to store the images
directory = Path('sign_images')

# Create the directory if it doesn't exist
directory.mkdir(parents=True, exist_ok=True)

# Define the path and filename of the CSV file
csv_file = 'signing_alphabet.csv'

# Create an empty list to store the image paths and labels
data = []

# Set up the video capture
cap = cv2.VideoCapture(0)

# Set up Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces, facial landmarks, gestures, and pose landmarks in the frame
    results = holistic.process(frame_rgb)

    # Overlay the facial landmarks on the frame
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Overlay the pose landmarks on the frame
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    # Overlay the hand landmarks on the frame
    if results.left_hand_landmarks or results.right_hand_landmarks:
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Signing Alphabet Recorder', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Exit the loop if the 'Esc' key is pressed
    if key == 27:  # 27 is the ASCII code for 'Esc'
        break

    # Record the image if a valid letter key is pressed
    for letter, label in signing_alphabet.items():
        if key == ord(letter):
            # Define the image path and label with directory
            image_path = directory / f'sign_{letter}_{len(data)}.jpg'
            image_label = letter

            # Save the image
            cv2.imwrite(str(image_path), frame)

            # Append the image path and label to the list
            data.append({'image_path': str(image_path), 'label': image_label})
            print(f'Saved image: {image_path}')
            break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()

# Create a DataFrame from the list
data = pd.DataFrame(data)

# Save the DataFrame to the CSV file
data.to_csv(csv_file, index=False)
print(f'Saved CSV file: {csv_file}')