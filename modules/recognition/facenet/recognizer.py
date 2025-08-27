# modules/recognition/FaceNet/facenet_recognizer.py
import torch
from typing import Union
from facenet_pytorch import InceptionResnetV1, MTCNN # InceptionResnetV1 as a facenet backbone
from PIL import Image
from torchvision import transforms
import numpy as np
import os 

class FaceNetRecognizer:
    """
    This service extracts facial embeddings using a pre-trained FaceNet model.
    """
    def __init__(self, device: str = 'cpu'):

        self.device = torch.device(device)
        print(f"Initializing FaceNetRecognizer on device: {self.device}")

        # pretrained='vggface2' o 'resnet' son opciones comunes. 'vggface2' suele ser popular.
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # MTCNN for face detection and alignment
        self.mtcnn = MTCNN(keep_all=False, device=self.device) 
        
        # FaceNet (InceptionResnetV1) espera entradas de 160x160 píxeles, normalizadas.
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)), 
            transforms.ToTensor(),        
            # Specific normalization for FaceNet
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Escala a [-1, 1]
        ])

    def get_embedding(self, image: np.ndarray) -> Union[torch.Tensor, None]:
        """
        Calculate the facial embedding for a given image input.
        
        Args:
            image: PIL Image (Image.Image), or an array NumPy (np.ndarray),
                         or image path (str).
        
        Returns:
            Pytorch Tensor with the facial embedding, or None if the input is invalid.
        """
        # Detect and align the face using MTCNN
        face = self.mtcnn(image)
        if face is None:
            print("No face detected in the image.")
            return None
        # Process the image with the defined transformations
        #image_tensor = self.transform(image).unsqueeze(0).to(self.device) 
        image_tensor = face.unsqueeze(0).to(self.device)
        
        # Calculate the embedding using the FaceNet model
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        # The embedding is a 512-dimensional vector
        return embedding.squeeze(0) # Delete the batch dimension for a single image embedding
    
    def get_distance(self, emb1: torch.Tensor, emb2: torch.Tensor, metric: str) -> float:
        """
        Calculate the distance between two facial embeddings.
        
        Args:
            emb1: First embedding tensor.
            emb2: Second embedding tensor.
            metric: Distance metric to use ('euclidean' or 'cosine').
        
        Returns:
            Distance as a float.
        """
        if emb1.shape != emb2.shape:
            raise ValueError("Embeddings must have the same shape.")
        
        # Calculate the distance
        if metric == 'cosine':
            # Cosine distance
            emb1_norm = emb1 / emb1.norm(p=2, dim=0, keepdim=True)
            emb2_norm = emb2 / emb2.norm(p=2, dim=0, keepdim=True)
            distance = 1 - torch.dot(emb1_norm, emb2_norm).item()
        elif metric == 'euclidean':
            # Euclidean distance
            distance = torch.norm(emb1 - emb2).item()
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
        
        return distance