import copy
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np
from numpy import typing as npt

from fingerspelling_to_pandas_singlehand_landmarks import (
    create_column_map,
    extract_hand_point_cloud,
)
from pipeline_fingerspelling5 import generate_hand_landmark_columns
from visualize_fingerspelling5_single_frame_cv2 import draw_hand as draw_hand_fancy


def map_values(val: float, size: int) -> int:
    return int(linear_map(val, -1, 1, 0, 1) * size)


def linear_map(value, a_min, a_max, n_min, n_max):
    # Perform linear mapping
    return n_min + (n_max - n_min) * (value - a_min) / (a_max - a_min)


def draw_hand_yz(canvas, landmarks):
    height, width, _ = canvas.shape
    # Landmark indices and edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
        (5, 9),
        (9, 13),
        (13, 17),
    ]
    if landmarks.shape != (21, 3):
        raise ValueError("Landmarks have incorrect shape.")
    values = landmarks[:, 2]
    cmin = -0.5  # min(values)
    cmax = 0.5  # max(values)

    # Function to interpolate points between two landmarks
    def interpolate_points(p1, p2, num_points):
        return np.linspace(p1, p2, num_points + 2)  # [1:-1]

    # Draw landmarks and edges
    for i in range(21):
        x, y = landmarks[i, [2, 1]]
        x = map_values(x, width)
        y = map_values(y, height)
        cv2.putText(
            canvas,
            str(i),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for edge in edges:
        x1, y1 = landmarks[edge[0], [2, 1]]
        x2, y2 = landmarks[edge[1], [2, 1]]
        x1_mapped = map_values(x1, width)
        x2_mapped = map_values(x2, width)
        y1_mapped = map_values(y1, height)
        y2_mapped = map_values(y2, height)

        cv2.line(
            canvas,
            (x1_mapped, y1_mapped),
            (x2_mapped, y2_mapped),
            (0, 0, 0),
            2,
        )

        num_interpolation = 1
        interpolated_x = interpolate_points(x1, x2, num_interpolation)
        interpolated_y = interpolate_points(y1, y2, num_interpolation)
        interpolated_values = interpolate_points(
            values[edge[0]], values[edge[1]], num_interpolation
        )

        for x, y, value in zip(interpolated_x, interpolated_y, interpolated_values):
            x_mapped = map_values(x, width)
            y_mapped = map_values(y, height)
            color_value = int(255 * (value - cmin) / (cmax - cmin))
            color_bgr = (color_value, color_value, color_value)  # BGR format
            cv2.circle(
                canvas,
                (x_mapped, y_mapped),
                3 + 1,
                (0, 255, 255),
                -1,
            )
            cv2.circle(
                canvas,
                (x_mapped, y_mapped),
                3,
                color_bgr,
                -1,
            )
    return canvas


def draw_hand_xz(canvas, landmarks):
    height, width, _ = canvas.shape
    # Landmark indices and edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
        (5, 9),
        (9, 13),
        (13, 17),
    ]
    if landmarks.shape != (21, 3):
        raise ValueError("Landmarks have incorrect shape.")
    values = landmarks[:, 2]
    cmin = -0.5  # min(values)
    cmax = 0.5  # max(values)

    # Function to interpolate points between two landmarks
    def interpolate_points(p1, p2, num_points):
        return np.linspace(p1, p2, num_points + 2)  # [1:-1]

    # Draw landmarks and edges
    for i in range(21):
        x, y = landmarks[i, [0, 2]]
        y = map_values(y, height)
        x = map_values(x, width)
        cv2.putText(
            canvas,
            str(i),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for edge in edges:
        x1, y1 = landmarks[edge[0], [0, 2]]
        x2, y2 = landmarks[edge[1], [0, 2]]
        y1_mapped = map_values(y1, height)
        y2_mapped = map_values(y2, height)
        x1_mapped = map_values(x1, width)
        x2_mapped = map_values(x2, width)

        cv2.line(
            canvas,
            (x1_mapped, y1_mapped),
            (x2_mapped, y2_mapped),
            (0, 0, 0),
            2,
        )

        num_interpolation = 1
        interpolated_x = interpolate_points(x1, x2, num_interpolation)
        interpolated_y = interpolate_points(y1, y2, num_interpolation)
        interpolated_values = interpolate_points(
            values[edge[0]], values[edge[1]], num_interpolation
        )

        for x, y, value in zip(interpolated_x, interpolated_y, interpolated_values):
            y_mapped = map_values(y, height)
            x_mapped = map_values(x, width)
            color_value = int(255 * (value - cmin) / (cmax - cmin))
            color_bgr = (color_value, color_value, color_value)  # BGR format
            cv2.circle(
                canvas,
                (x_mapped, y_mapped),
                3 + 1,
                (0, 255, 255),
                -1,
            )
            cv2.circle(
                canvas,
                (x_mapped, y_mapped),
                3,
                color_bgr,
                -1,
            )
    return canvas


def number_to_letter(number: int) -> str:
    if number < 0:
        raise ValueError("Input number must be non-negative.")
    if number <= 8:
        return chr(ord("a") + number)
    elif number <= 24:
        return chr(ord("a") + number + 1)
    else:
        raise ValueError("Invalid input: number out of range.")


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    if landmarks.shape != (21, 3):
        raise ValueError("Landmarks have incorrect shape.")
    for i in range(21):
        x, y = int(landmarks[i, 0] * width), int(landmarks[i, 1] * height)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    return frame


def draw_xz(size: int, landmarks) -> npt.NDArray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    canvas = draw_hand_xz(canvas, landmarks)
    return canvas


def draw_yz(size: int) -> npt.NDArray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    return canvas


def create_canvas(size: int) -> npt.NDArray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    return canvas


def get_xy_range(
    landmark_coords: npt.NDArray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if landmark_coords.shape != (21, 3):
        raise ValueError

    x_min, y_min = np.min(landmark_coords, axis=0)[:2]
    x_max, y_max = np.max(landmark_coords, axis=0)[:2]
    return ((x_min, x_max), (y_min, y_max))


def crop_image_with_bounds(image, x_min, y_min, x_max, y_max):
    # Get image dimensions
    m, n, _ = image.shape

    # Clip coordinates to ensure they're within the image bounds
    x_min = max(0, min(x_min, n))
    y_min = max(0, min(y_min, m))
    x_max = max(0, min(x_max, n))
    y_max = max(0, min(y_max, m))

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


def resize_and_pad(image, target_height, target_width):
    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]

    # Create a blank canvas with the target dimensions
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the aspect ratio of the original image
    try:
        aspect_ratio = original_width / original_height
    except ZeroDivisionError:
        return padded_image

    # Calculate the new dimensions while maintaining the aspect ratio
    if target_width / aspect_ratio <= target_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate the position to paste the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the resized image onto the canvas
    padded_image[
        y_offset : y_offset + new_height, x_offset : x_offset + new_width
    ] = resized_image

    return padded_image


def crop_hand(frame, point_coords):
    buffer = 0.15
    height, width, _ = frame.shape
    (x_min, x_max), (y_min, y_max) = get_xy_range(point_coords)
    x_min, x_max = int(x_min * (1 - buffer) * width), int(x_max * (1 + buffer) * width)
    y_min, y_max = int(y_min * (1 - buffer) * height), int(
        y_max * (1 + buffer) * height
    )
    cropped_image = crop_image_with_bounds(
        image=frame, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
    )
    resized_padded_image = resize_and_pad(
        cropped_image, target_height=height, target_width=width
    )
    return resized_padded_image


def column_map_to_point_cloud(column_map: Dict[str, float]) -> npt.NDArray:
    point_cloud = np.array(
        [column_map[col_name] for col_name in landmark_columns], dtype=np.float32
    )
    return np.reshape(point_cloud, (-1, 3))


def normalize_point_cloud(point_cloud: npt.NDArray) -> npt.NDArray:
    point_mean = np.mean(point_cloud, axis=-2, keepdims=True)
    point_cloud = point_cloud - point_mean
    point_scale = (1 / np.max(np.abs(point_cloud))) * 0.999999
    return point_cloud * point_scale


def extract_hand_landmarks(
    hand_result: mp.solutions.hands.Hands,
) -> npt.NDArray:
    """
    VERY similar to
    extract_hand_point_cloud from fingerspelling_to_pandas_singlehand_landmarks
    """
    landmarks = np.full((21, 3), np.nan, dtype=np.float32)
    if hand_result.multi_hand_landmarks:
        if len(hand_result.multi_hand_landmarks) > 1:
            raise ValueError("More than hand detected. Configure mp.Hands properly.")
        # Fill landnmarks with x,y,z values.
        for hand_landmarks in hand_result.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmarks[i, :] = (landmark.x, landmark.y, landmark.z)
    return landmarks


def transform_hand_results(results) -> npt.NDArray:
    hand_point_cloud = extract_hand_point_cloud(results)
    column_map = create_column_map(hand_point_cloud)

    point_cloud = column_map_to_point_cloud(column_map)
    return np.nan_to_num(point_cloud)


if __name__ == "__main__":
    # Initialize the MediaPipe solutions
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    mp_hands_cropped = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )

    # Create a VideoCapture object to capture video from the webcam
    video_capture = cv2.VideoCapture(
        0
    )  # 0 indicates the default webcam, change it to the appropriate index if you have multiple webcams

    landmark_columns = generate_hand_landmark_columns()

    # Loop to continuously read frames from the webcam
    while True:
        # Read the current frame from the webcam
        ret, frame = video_capture.read()
        frame_raw = copy.deepcopy(frame)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces, facial landmarks, gestures, and pose landmarks in the frame
        results = mp_hands.process(frame_rgb)
        point_cloud = transform_hand_results(results)

        point_cloud_raw = point_cloud
        point_coords = normalize_point_cloud(point_cloud)

        canvas_xz = create_canvas(frame.shape[0])
        canvas_yz = create_canvas(frame.shape[0])
        landmarks = extract_hand_landmarks(results)
        if not np.any(np.isnan(landmarks)):
            draw_hand_fancy(frame, landmarks)
            canvas_xz = draw_hand_xz(canvas_xz, point_coords)
            canvas_yz = draw_hand_yz(canvas_yz, point_coords)
        output_frame = np.concatenate([canvas_xz, frame, canvas_yz], axis=1)

        # create cropped/zoomed comparison canvas
        cropped_hand = crop_hand(frame=frame_raw, point_coords=point_cloud_raw)
        cropped_hand_rgb = crop_hand(frame=frame_rgb, point_coords=point_cloud_raw)

        results_cropped = mp_hands_cropped.process(cropped_hand_rgb)
        point_cloud_cropped = transform_hand_results(results_cropped)
        point_cloud_cropped = normalize_point_cloud(point_cloud_cropped)
        canvas_xz = create_canvas(frame.shape[0])
        canvas_yz = create_canvas(frame.shape[0])
        landmarks_cropped = extract_hand_landmarks(results_cropped)
        if not np.any(np.isnan(landmarks_cropped)):
            draw_hand_fancy(cropped_hand, landmarks_cropped)
            canvas_xz = draw_hand_xz(canvas_xz, point_cloud_cropped)
            canvas_yz = draw_hand_yz(canvas_yz, point_cloud_cropped)
        compare_canvas = np.concatenate([canvas_xz, cropped_hand, canvas_yz], axis=1)

        output_frame = np.concatenate([output_frame, compare_canvas], axis=0)

        # show additional information
        _, output_width, _ = output_frame.shape

        bottom_canvas_height = 100

        bottom_canvas = (
            np.ones((bottom_canvas_height, output_width, 3), dtype=np.uint8) * 255
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        y_offset = 25
        font_size = 1

        x_diff = point_coords[12, 0] - point_coords[8, 0]
        text = f"(x_12 - x_8): {x_diff:.8f}"
        cv2.putText(
            bottom_canvas,
            text,
            (100, 20),
            font,
            font_scale,
            font_color,
            font_size,
        )

        z_diff = point_coords[12, 2] - point_coords[8, 2]
        text = f"(z_12 - z_8): {z_diff:.8f}"
        cv2.putText(
            bottom_canvas,
            text,
            (100, 40),
            font,
            font_scale,
            font_color,
            font_size,
        )

        output_frame = np.concatenate([output_frame, bottom_canvas], axis=0)

        # resize image
        scaling_factor = 0.65
        img_height, img_width, _ = output_frame.shape
        scale_height = int(img_height * scaling_factor)
        scale_width = int(img_width * scaling_factor)
        output_frame = cv2.resize(output_frame, (scale_width, scale_height))

        # Display the frame in a window called "Webcam Feed"
        cv2.imshow("Webcam Feed", output_frame)

        # Wait for the 'q' key to be pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object and close the window
    video_capture.release()
    cv2.destroyAllWindows()
