import cv2
import numpy as np
from utils.feature_extraction import extract_embedding
from utils.recognition import is_same_person
import json
from PIL import Image
import matplotlib.pyplot as plt

# Load reference embeddings from reference_list.json
with open("embeddings/reference_list.json") as f:
    reference_data = json.load(f)

# Convert reference embeddings to numpy arrays and ensure correct shape
for person in reference_data:
    reference_data[person] = np.array(reference_data[person])

# Load OpenCV's pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture using OpenCV
cap = cv2.VideoCapture(0)

print("Press 'Ctrl+C' to stop the program.")

# Setup for displaying video with matplotlib
plt.ion()  # Turn on interactive mode for live updates

try:
    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale (required by face detector)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face region from the frame
            face_region = frame[y:y + h, x:x + w]
            face_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))

            # Extract embedding for the detected face
            current_embedding = extract_embedding(face_image)

            # Ensure current_embedding is a 1D array
            current_embedding = np.squeeze(current_embedding)

            # Check against all saved reference embeddings
            recognized = False
            for person, ref_embedding in reference_data.items():
                # Ensure reference embedding is a 1D array
                ref_embedding = np.squeeze(ref_embedding)

                if is_same_person(ref_embedding, current_embedding):
                    # If recognized, draw the name of the person
                    cv2.putText(frame, f"Recognized: {person}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    recognized = True
                    break

            if recognized:
                # Logic for recognized face, e.g., additional display options
                pass

        # Convert the frame to RGB for matplotlib display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with matplotlib
        plt.imshow(frame_rgb)
        plt.axis('off')  # Turn off axis
        plt.draw()
        plt.pause(0.001)  # Pause to update the display

except KeyboardInterrupt:
    print("Stopped video stream.")

finally:
    cap.release()
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the last frame before closing
