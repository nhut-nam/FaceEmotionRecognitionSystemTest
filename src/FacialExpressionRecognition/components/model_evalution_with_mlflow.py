import torch
from FacialExpressionRecognition.entity.config_entity import EvaluationConfig
import mlflow
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from urllib.parse import urlparse
import mlflow.pytorch
from FacialExpressionRecognition.models.resnet34 import get_resnet34_model
from FacialExpressionRecognition.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(self):
        # Load model
        self.model = self.load_model(model_path=self.config.path_of_model, params=self.config.all_params)

        # Data transformations
        data_transforms = transforms.Compose([
            transforms.Resize((self.config.params_image_size[0], self.config.params_image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load dataset
        train_dataset = ImageFolder(root=self.config.data_path + "/FER2013/train", transform=data_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)

        val_dataset = ImageFolder(root=self.config.data_path + "/FER2013/val", transform=data_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.params_batch_size, shuffle=False)

        test_dataset = ImageFolder(root=self.config.data_path + "/FER2013/test", transform=data_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False)

        # Evaluate model
        train_accuracy, train_loss = self.evaluate_score(train_dataloader)
        val_accuracy, val_loss = self.evaluate_score(val_dataloader)
        test_accuracy, test_loss = self.evaluate_score(test_dataloader)

        print(f'Accuracy of the model on the training dataset: {train_accuracy:.2f}%')
        print(f'Accuracy of the model on the validation dataset: {val_accuracy:.2f}%')
        print(f'Accuracy of the model on the test dataset: {test_accuracy:.2f}%')

        self.metrics = {
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss
        }

        self.save_score()

    def evaluate_score(self, data_loader):
        correct = 0
        total = 0
        losses = []
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                losses.append(loss.item())

        accuracy = 100 * correct / total
        return accuracy, sum(losses) / len(losses)

    @staticmethod
    def load_model(model_path, params):
        model = get_resnet34_model(pretrained=False, num_classes=params['NUM_CLASSES'])
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint)
        
        model.eval()
        return model

    def log_into_mlflow(self):
        import dagshub
        REPO_OWNER = 'nhut-nam'
        REPO_NAME = 'FaceEmotionRecognitionSystemTest'
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            for key, value in self.metrics.items():
                mlflow.log_metric(key, value)

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name=self.config.all_params['MODEL_NAME'])
            else:
                mlflow.pytorch.log_model(self.model, "model")

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.metrics)

    