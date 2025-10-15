import numpy as np
import torch
import os
# from FacialExpressionRecognition.models.resnet34 import get_resnet34_model
from FacialExpressionRecognition.models.emonext import get_model 
from PIL import Image
from torchvision import transforms

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = get_model(num_classes=7)
        model.load_state_dict(torch.load(os.path.join("artifacts/training", "model.pth")))
        model.eval()

        image_name = self.filename
        with torch.no_grad():
            image = self.load_image(image_name)
            image = self.preprocess_image(image)  # (1, 10, 3, H, W)

            # Giống test_model: view và average 10 crop
            bs, ncrops, c, h, w = image.shape
            image = image.view(-1, c, h, w)

            output = model(image)
            output_avg = output.view(bs, ncrops, -1).mean(1)

            probs = torch.nn.functional.softmax(output_avg, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs, dim=1).values.item()

        return prediction, confidence


    def load_image(self, image_path):
        image = Image.open(image_path).convert('L')
        return image

    def preprocess_image(self, image):
        transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(140),
            transforms.TenCrop(64),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])
            ),
        ])
        return transform(image).unsqueeze(0)