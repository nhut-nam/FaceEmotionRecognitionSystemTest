import os
import torch
import torchvision.models as models
import torchvision
import tqdm
from zipfile import ZipFile
from FacialExpressionRecognition import logger
from FacialExpressionRecognition.entity.config_entity import TrainingConfig
from FacialExpressionRecognition.models.resnet34 import get_resnet34_model

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_model(self):
        if self.config.model_name == "resnet34":
            self.model = get_resnet34_model(num_classes=self.config.params_num_classes)
            self.model.load_state_dict(torch.load(self.config.updated_base_model_path))
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.config.params_num_classes)
            self.model.eval()
        else:
            raise ValueError(f"Model {self.config.model_name} not supported.")
        

    def extract_dataset(self):
        with ZipFile(self.config.training_data_path, 'r') as zip_ref:
            os.makedirs(self.config.dataset_path, exist_ok=True)
            zip_ref.extractall(self.config.dataset_path)

    def train(self):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.config.params_image_size[0], self.config.params_image_size[1])),
            torchvision.transforms.ToTensor()
        ])

        self.extract_dataset()
        
        train_data = torchvision.datasets.ImageFolder(root=self.config.dataset_path, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config.params_batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_model()
        self.model = self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)

        num_epochs = self.config.params_epochs

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm.tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        Training.save_model(self.model, self.config.trained_model_path)

    @staticmethod
    def save_model(model, path):
        torch.save(model, path)
        logger.info(f"Model saved at {path}")