# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import cv2

# model = models.resnet50(weights='IMAGENET1K_V1')
# model.eval()  

# def extract_embedding(frame):
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     image = transform(image).unsqueeze(0)  

#     with torch.no_grad():
#         embedding = model(image).squeeze().cpu().numpy()  

#     return embedding

from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np

# Load the pre-trained ResNet50 model and remove the final layer
model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final fully connected layer

def extract_embedding(frame):
    image = frame.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        embedding = model(image).squeeze().cpu().numpy()

    return embedding.flatten()  # Flatten to ensure 1-D array
