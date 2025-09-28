import torch
import torchvision.models as models

def get_resnet34_model(pretrained=True, num_classes=7):
    model = models.resnet34(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model