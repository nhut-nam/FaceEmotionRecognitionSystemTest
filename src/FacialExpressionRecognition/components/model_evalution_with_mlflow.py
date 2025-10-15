import torch
import os
from FacialExpressionRecognition.entity.config_entity import EvaluationConfig
import mlflow
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from urllib.parse import urlparse
import mlflow.pytorch
from FacialExpressionRecognition.models.resnet34 import get_resnet34_model
from FacialExpressionRecognition.models.emonext import get_model 
from FacialExpressionRecognition.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        # Load model
        self.model = self.load_model(model_path=self.config.path_of_model, params=self.config.all_params)
        self.model.to(self.device)
        self.model.eval()

        # Data transformations
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(140),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(self.config.params_image_size[0]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        val_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(140),
            transforms.RandomCrop(self.config.params_image_size[0]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(140),
            transforms.TenCrop(self.config.params_image_size[0]),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack([crop.repeat(3, 1, 1) for crop in crops])
            ),
        ])

        # Load datasets
        base_path = os.path.join(self.config.data_path, "FER2013")
        train_dataset = ImageFolder(root=os.path.join(base_path, "train"), transform=train_transforms)
        val_dataset = ImageFolder(root=os.path.join(base_path, "val"), transform=val_transforms)
        test_dataset = ImageFolder(root=os.path.join(base_path, "test"), transform=test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.params_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False)

        # Evaluate
        train_accuracy, train_loss = self.evaluate_score(train_dataloader)
        val_accuracy, val_loss = self.evaluate_score(val_dataloader)
        test_accuracy, test_loss = self.evaluate_score(test_dataloader)

        print(f"Accuracy of the model on the training dataset: {train_accuracy:.2f}%")
        print(f"Accuracy of the model on the validation dataset: {val_accuracy:.2f}%")
        print(f"Accuracy of the model on the test dataset: {test_accuracy:.2f}%")

        self.metrics = {
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        }

        # Save metrics
        self.save_score()
        # Optional: log lên MLflow
        # self.log_into_mlflow()


    def evaluate_score(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Kiểm tra xem input có TenCrop hay không (batch, ncrops, C, H, W)
                if images.ndim == 5:
                    bs, ncrops, c, h, w = images.shape
                    images = images.view(-1, c, h, w)
                    with torch.autocast(self.device.type):
                        logits = self.model(images)
                    outputs = logits.view(bs, ncrops, -1).mean(1)  # trung bình qua các crop
                else:
                    with torch.autocast(self.device.type):
                        outputs = self.model(images)

                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / total
        return accuracy, avg_loss

    @staticmethod
    def load_model(model_path, params):
        if params['MODEL_NAME'] == 'resnet34':
            model = get_resnet34_model(pretrained=False, num_classes=params['NUM_CLASSES'])
        elif params['MODEL_NAME'] == 'emonext':
            model = get_model(num_classes=params['NUM_CLASSES'])

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
        model.eval()
        return model

    def log_into_mlflow(self):
        import dagshub
        import mlflow
        from urllib.parse import urlparse

        REPO_OWNER = 'nhut-nam'
        REPO_NAME = 'FaceEmotionRecognitionSystemTest'

        # Khởi tạo kết nối với DagsHub (đã tự set tracking và registry URI)
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            for key, value in self.metrics.items():
                mlflow.log_metric(key, value)

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    self.model,
                    "model",
                    # registered_model_name=self.config.all_params["MODEL_NAME"]
                )
            else:
                mlflow.pytorch.log_model(self.model, "model")


    def save_score(self):
        save_json(path=Path("scores.json"), data=self.metrics)

    