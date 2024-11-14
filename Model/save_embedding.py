# import numpy as np
# from utils.feature_extraction import extract_embedding
# from PIL import Image
# import json

# with open("embeddings/reference_list.json", "r") as f:
#     reference_data = json.load(f)

# def save_reference_embedding(name, image_path):
#     image = Image.open(image_path)
#     embedding = extract_embedding(np.array(image))

#     embedding_path = f"embeddings/{name}_reference.npy"
#     np.save(embedding_path, embedding)

#     reference_data[name] = embedding_path
#     with open("embeddings/reference_list.json", "w") as f:
#         json.dump(reference_data, f)
#     print(f"Reference embedding for {name} saved successfully.")

# save_reference_embedding("Person3", "./uploads/upimg3.jpg")

import numpy as np
from utils.feature_extraction import extract_embedding
from PIL import Image
import json

# Load existing reference data
with open("embeddings/reference_list.json", "r") as f:
    reference_data = json.load(f)

def save_reference_embedding(name, image_path):
    # Open the image as a PIL Image (not as a numpy array)
    image = Image.open(image_path)
    
    # Extract the embedding from the PIL Image (not a numpy array)
    embedding = extract_embedding(image)  # Pass PIL Image directly, not np.array(image)

    # Save the embedding directly in the JSON file as a list (JSON compatible)
    reference_data[name] = embedding.tolist()  # Convert embedding to a list for JSON compatibility

    with open("embeddings/reference_list.json", "w") as f:
        json.dump(reference_data, f)
    print(f"Reference embedding for {name} saved successfully.")

# Example: Save the embedding for "Person3"
save_reference_embedding("Person1", "./uploads/uploaded_image.jpg")
