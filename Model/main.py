import cv2
import numpy as np
import json
from utils.feature_extraction import extract_embedding
from utils.recognition import is_same_person

with open("embeddings/reference_list.json", "r") as f:
    reference_data = json.load(f)

reference_embeddings = {
    name: np.load(embedding_path)
    for name, embedding_path in reference_data.items()
}

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        current_embedding = extract_embedding(face)

        matched_person = None
        for name, reference_embedding in reference_embeddings.items():
            if is_same_person(reference_embedding, current_embedding):
                matched_person = name
                break

        if matched_person:
            cv2.putText(frame, f"{matched_person} Recognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
