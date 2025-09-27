import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torchvision.models as models
import torchvision
from FacialExpressionRecognition import logger
from FacialExpressionRecognition.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_model(self, pretrained=True):
        model = torchvision.models.resnet34(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, self.config.params_classes)
        logger.info(f"Model: {model} - Parameters: {sum(p.numel() for p in model.parameters())}")
        return model


    def update_base_model(self):
        model = self.get_model(pretrained=True)
        self.save_model(model, self.config.updated_base_model_path)

    @staticmethod
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)