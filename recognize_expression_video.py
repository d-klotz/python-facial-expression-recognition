import cv2
import os
import numpy as np
from tqdm import tqdm
import face_recognition
from deepface import DeepFace


def load_images_from_folder(folder_path):
    known_face_encodings = []
    known_face_names = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' doesn't exist. Creating it...")
        os.makedirs(folder_path)
        return known_face_encodings, known_face_names

    for filename in os.listdir(folder_path):
        if (filename.endswith(".jpg") or filename.endswith(".png")):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)

            if face_encoding:  # Changed from face_recognition to face_encoding
                face_encoding = face_encoding[0]
                name = os.path.splitext(filename)[0]  # Removed the [:-1] slice
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

def detect_faces_and_emotions(video_path, output_path, known_face_encodings, known_face_names):
    # Load the video from the video path
    cap = cv2.VideoCapture(video_path)

    # check if video can be opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # define codec and create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop to process each frame with a progress bar
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze the frame to detect faces and emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the location of the face known in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Initialize a list of names for each face detected
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)

        # Iterate over each face detected by DeepFace
        for result in result:
            # Get the bounding box of the face
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']

            # Get dominant emotion
            dominant_emotion = result['dominant_emotion']

            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Associate the known face with the detected face
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if x <= left <= x+w and y <= top <= y+h:
                    # Write the name under the face
                    cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Write the frame to the output video
        out.write(frame)
        
    # Release the video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

image_folder = "images"
known_face_encodings, known_face_names = load_images_from_folder(image_folder)

script_dir = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
input_video_path = os.path.join(script_dir, 'new-video.mp4')
output_video_path = os.path.join(script_dir, 'output_video_recognized.mp4')

detect_faces_and_emotions(input_video_path, output_video_path, known_face_encodings, known_face_names)

