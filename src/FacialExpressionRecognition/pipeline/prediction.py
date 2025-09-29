import numpy as np
import torch
import os
from FacialExpressionRecognition.models.resnet34 import get_resnet34_model
from PIL import Image
from torchvision import transforms

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = get_resnet34_model(pretrained=False, num_classes=7)
        model.load_state_dict(torch.load(os.path.join("artifacts/training", "model.pth")))
        model.eval()

        # Load and preprocess the image
        image_name = self.filename
        with torch.no_grad():
            image = self.load_image(image_name)
            image = self.preprocess_image(image)

            output = model(image)

            # Softmax để lấy xác suất
            probs = torch.nn.functional.softmax(output, dim=1)

            # Lấy class dự đoán và xác suất cao nhất
            prediction = torch.argmax(probs, dim=1).item()      # index class
            confidence = torch.max(probs, dim=1).values.item()  # xác suất

        return prediction, confidence

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0)