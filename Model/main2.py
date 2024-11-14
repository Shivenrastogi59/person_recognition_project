import insightface
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)

dataset_dir = "./dataset"
embeddings = []
image_paths = []

def get_face_embedding_and_bbox(image_path):
    image = cv2.imread(image_path)
    faces = model.get(image)
    if faces:
        face = faces[0]
        embedding = face.embedding
        bbox = face.bbox.astype(int)
        return embedding, bbox
    return None, None

for img_file in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_file)
    embedding, bbox = get_face_embedding_and_bbox(img_path)
    if embedding is not None:
        embeddings.append(embedding)
        image_paths.append((img_path, bbox))

embeddings = np.array(embeddings)

def display_image_with_bounding_box(image_path, bounding_box):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    x, y, w, h = bounding_box
    box_margin = int(0.1 * h)  

    x += box_margin
    y += box_margin
    w -= 2 * box_margin
    h -= 2 * box_margin
    
    cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def search_target_image(target_image_path, k=5):
    target_embedding, target_bbox = get_face_embedding_and_bbox(target_image_path)
    if target_embedding is None:
        print("No face found in the target image.")
        return []
    
    distances = np.linalg.norm(embeddings - target_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    
    results = [(image_paths[i][0], image_paths[i][1], distances[i]) for i in nearest_indices]
    return results

target_image_path = "./uploads/uploaded_image.jpg"

results = search_target_image(target_image_path, k=5)

for img_path, bbox, distance in results:
    print(f"Match: {img_path} with distance: {distance}")
    display_image_with_bounding_box(img_path, bbox)
