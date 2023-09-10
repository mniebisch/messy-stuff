import copy

import cv2
import pandas as pd

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for default camera, change if needed

# Create an empty list to store the data
data = []


def record_sign(letter, individual_id):
    global data

    # Wait for spacebar to start recording
    while True:
        ret, frame = cap.read()
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            f"Press spacebar to start recording: {letter.upper()}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(f"Recording {letter.upper()} for Individual {individual_id}", frame)

        # Break the loop if spacebar is pressed
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    # Start recording the movement
    frames = []
    while True:
        ret, frame = cap.read()
        frames.append(frame)

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        frame_show = copy.deepcopy(frame)

        # Add text to the frame
        cv2.putText(
            frame_show,
            f"Recording letter: {letter.upper()}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(f"Recording {letter} for Individual {individual_id}", frame_show)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Convert frames to a video and store the file path as .mp4
    video_path = f"{individual_id}_{letter}.mp4"
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (frame.shape[1], frame.shape[0]),
    )
    for frame in frames:
        out.write(frame)
    out.release()

    # Close window for the letter
    cv2.destroyAllWindows()

    # Store data in the list
    data.append(
        {"Individual_ID": individual_id, "Sign": letter, "Movement": video_path}
    )


# Main loop to record individuals and alphabet
while True:
    individual_id = input("Enter individual ID (or 'exit' to finish): ")
    if individual_id.lower() == "exit":
        break

    for letter in "abcdefghiklmnopqrstuvwxyz":
        record_sign(letter, individual_id)

# Release the camera
cap.release()

# Convert the list of dictionaries to a dataframe
df = pd.DataFrame(data)

# Save the dataframe to a CSV file
df.to_csv("labeled_dataset.csv", index=False)
