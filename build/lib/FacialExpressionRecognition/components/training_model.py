import os
import sys
import torch
import torchvision.models as models
import torchvision
from tqdm import tqdm
import numpy as np
from zipfile import ZipFile
from FacialExpressionRecognition import logger
from FacialExpressionRecognition.entity.config_entity import TrainingConfig
from FacialExpressionRecognition.models.resnet34 import get_resnet34_model
from FacialExpressionRecognition.models.emonext import get_model 
import numpy as np
from torch.cuda.amp import GradScaler

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.early_stopping_patience = 12
        self.best_val_accuracy = 0
        self.scaler = GradScaler(enabled=False)
    
    def get_model(self):
        if self.config.model_name == "resnet34":
            self.model = get_resnet34_model(num_classes=self.config.params_num_classes)
            self.model.load_state_dict(torch.load(self.config.updated_base_model_path))
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.config.model_name == "emonext":
            self.model = get_model(num_classes=self.config.params_num_classes)
            self.model.eval()
        else:
            raise ValueError(f"Model {self.config.model_name} not supported.")
        

    def extract_dataset(self):
        with ZipFile(self.config.training_data_path, 'r') as zip_ref:
            os.makedirs(self.config.dataset_path, exist_ok=True)
            zip_ref.extractall(self.config.dataset_path)

    def get_optimizer(self):
        if self.config.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        elif self.config.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate, momentum=0.9)
        elif self.config.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.params_learning_rate)
        else:
            raise ValueError(f"Optimizer {self.config.optimizer_name} not supported.")
        return optimizer

    def train(self):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(140),
            torchvision.transforms.RandomRotation(degrees=20),
            torchvision.transforms.RandomCrop(self.config.params_image_size[0]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(140),
            torchvision.transforms.RandomCrop(self.config.params_image_size[0]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(140),
            torchvision.transforms.TenCrop(self.config.params_image_size[0]),
            torchvision.transforms.Lambda(
                lambda crops: torch.stack(
                    [torchvision.transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            torchvision.transforms.Lambda(
                lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])
            ),
        ])

        self.extract_dataset()
        
        train_data = torchvision.datasets.ImageFolder(root=self.config.dataset_path / "FER2013" / "train", transform=train_transforms)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config.params_batch_size, shuffle=True)

        val_data = torchvision.datasets.ImageFolder(root=self.config.dataset_path / "FER2013" / "val", transform=val_transforms)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)

        test_data = torchvision.datasets.ImageFolder(root=self.config.dataset_path / "FER2013" / "test", transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_model()
        self.model = self.model.to(self.device)


        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self.get_optimizer()

        num_epochs = self.config.params_epochs

        counter = 0
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            if val_accuracy > self.best_val_accuracy:
                self.save_model(self.model, self.config.trained_model_path)
                counter = 0
                self.best_val_accuracy = val_accuracy
            else:
                counter += 1
                if counter >= self.early_stopping_patience:
                    print(
                        "Validation loss did not improve for %d epochs. Stopping training."
                        % self.early_stopping_patience
                    )
                    break

            self.test_model()


    def train_epoch(self):
        self.model.train()

        avg_accuracy = []
        avg_loss = []
        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.train_loader))
        for batch_idx, data in enumerate(self.train_loader):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type):
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels, label_smoothing=0.2)

            self.scaler.scale(loss).backward()      # scale loss trước khi backward
            self.scaler.step(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()

            predictions = torch.argmax(outputs, dim=1)
            batch_accuracy = (predictions == labels).sum().item() / labels.size(0)
            avg_accuracy.append(batch_accuracy)
            avg_loss.append(loss.item())
            pbar.set_postfix({"loss": np.mean(avg_loss), "accuracy": np.mean(avg_accuracy) * 100.0})
            pbar.update(1)

        pbar.close()
        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0
    
    def val_epoch(self):
        self.model.eval()

        avg_loss = []
        predicted_labels = []
        true_labels = []

        pbar = tqdm(
            unit="batch", file=sys.stdout, total=len(self.val_loader)
        )

        for batch_idx, (inputs, labels) in enumerate(self.val_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type):
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels, label_smoothing=0.2)
            
            predictions = torch.argmax(outputs, dim=1)
            avg_loss.append(loss.item())
            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()
        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        return np.mean(avg_loss), accuracy * 100.0
    
    def test_model(self):

        predicted_labels = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.test_loader))
        for batch_idx, (inputs, labels) in enumerate(self.test_loader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type):
                logits = self.model(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(1)
            predictions = torch.argmax(outputs_avg, dim=1)

            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update(1)

        pbar.close()

        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        print("Test Accuracy: %.4f %%" % (accuracy * 100.0))


    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved at {path}")