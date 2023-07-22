import cv2
import dlib

# Initialize the dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download the predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Create a VideoCapture object to capture video from the webcam
video_capture = cv2.VideoCapture(0)  # 0 indicates the default webcam, change it to the appropriate index if you have multiple webcams

# Loop to continuously read frames from the webcam
while True:
    # Read the current frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Determine the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Iterate over the facial landmarks and draw them on the frame
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the frame in a window called "Webcam Feed"
    cv2.imshow("Webcam Feed", frame)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
video_capture.release()
cv2.destroyAllWindows()