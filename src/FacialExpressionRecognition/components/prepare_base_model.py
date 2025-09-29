import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torchvision.models as models
import torchvision
from FacialExpressionRecognition import logger
from FacialExpressionRecognition.entity.config_entity import PrepareBaseModelConfig
from FacialExpressionRecognition.models.resnet34 import get_resnet34_model

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_model(self, pretrained=True):
        if self.config.params_model_name == "resnet34":
            self.model = get_resnet34_model(pretrained=pretrained, num_classes=self.config.params_classes)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.config.params_classes)
            self.model.eval()
            return self.model
        else:
            raise ValueError(f"Model {self.config.params_model_name} not supported.")


    def update_base_model(self):
        model = self.get_model(pretrained=True)
        self.save_model(model, self.config.updated_base_model_path)

    @staticmethod
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)