from FacialExpressionRecognition.constants import *
from FacialExpressionRecognition.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig
from FacialExpressionRecognition.utils.common import read_yaml, create_directories
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(os.path.join(config.base_model_path)),
            updated_base_model_path=Path(os.path.join(config.updated_base_model_path)),
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_model_name=params.MODEL_NAME,
            params_optimizer=params.OPTIMIZER,
            params_classes=params.NUM_CLASSES,
        )
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training
        artifacts_root = self.config.artifacts_root
        trained_model_path = os.path.join(training_config.trained_model_path)
        updated_base_model_path = os.path.join(self.config.prepare_base_model.updated_base_model_path)
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "data.zip")
        dataset_path = os.path.join(training_config.dataset_path)

        create_directories([Path(training_config.root_dir)])

        params = self.params
        params_epochs = params.EPOCHS
        params_batch_size = params.BATCH_SIZE
        params_is_augmentation = params.AUGMENTATION
        params_learning_rate = params.LEARNING_RATE
        params_num_classes = params.NUM_CLASSES
        params_image_size = params.IMAGE_SIZE
        params_model_name = params.MODEL_NAME

        training_config = TrainingConfig(   
            root_dir=Path(artifacts_root),
            trained_model_path=Path(trained_model_path),
            updated_base_model_path=Path(updated_base_model_path),
            training_data_path=Path(training_data),
            dataset_path=Path(dataset_path),
            model_name=params_model_name,
            params_epochs=params_epochs,
            params_batch_size=params_batch_size,
            params_is_augmentation=params_is_augmentation,
            params_learning_rate=params_learning_rate,
            params_num_classes=params_num_classes,
            params_image_size=params_image_size
        )

        return training_config