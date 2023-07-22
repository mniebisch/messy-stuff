import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "effective-octo-potato" / "data"

# Read the train.csv file
train_csv_path = data_dir / 'train.csv'  # Provide the path to the train.csv file
train_df = pd.read_csv(train_csv_path)

# Create an empty DataFrame to store the file table
file_table = pd.DataFrame(columns=['file_path', 'participant_id', 'sequence_id', 'frame_number'])

# Iterate over each row in the train.csv file
for _, row in train_df.iterrows():
    landmark_file_path = data_dir / row['path']
    participant_id = str(row['participant_id'])  # Convert participant_id to string
    sequence_id = str(row['sequence_id'])  # Convert sequence_id to string

    # Create a directory for the participant if it doesn't exist
    participant_dir = Path('data') / 'train_landmark_files' / participant_id
    participant_dir.mkdir(parents=True, exist_ok=True)

    # Create a directory for the sequence if it doesn't exist
    sequence_dir = participant_dir / sequence_id
    sequence_dir.mkdir(parents=True, exist_ok=True)

    # Read the landmark file
    landmark_df = pd.read_parquet(landmark_file_path)

    # Group the dataframe by frame number
    frames = landmark_df.groupby('frame')

    # Iterate over each frame
    for frame_number, frame_data in frames:
        # Create a separate file for each frame
        frame_file_path = sequence_dir / f'frame_{frame_number}.csv'

        # Save the frame data to the file
        frame_data.to_csv(frame_file_path, index=False)

        # Append the file information to the file table
        row_data = {
            'file_path': frame_file_path,
            'participant_id': participant_id,
            'sequence_id': sequence_id,
            'frame_number': frame_number
        }
        file_table = pd.concat([file_table, pd.DataFrame(row_data, index=[0])], ignore_index=True)

# Save the file table as a CSV file
file_table.to_csv('file_table.csv', index=False)