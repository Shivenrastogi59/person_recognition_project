# import numpy as np
# from scipy.spatial.distance import cosine

# def is_same_person(reference_embedding, current_embedding, threshold=0.65):
#     similarity = 1 - cosine(reference_embedding, current_embedding)
#     return similarity > threshold

# utils/recognition.py



import numpy as np
from scipy.spatial.distance import cosine

def is_same_person(reference_embedding, current_embedding):
    # Ensure that the embeddings are 1D arrays
    reference_embedding = np.squeeze(reference_embedding)
    current_embedding = np.squeeze(current_embedding)

    # Check that both embeddings have the same shape
    if reference_embedding.shape != current_embedding.shape:
        raise ValueError(f"Embedding shapes do not match: {reference_embedding.shape} vs {current_embedding.shape}")

    # Calculate similarity
    similarity = 1 - cosine(reference_embedding, current_embedding)
    return similarity
