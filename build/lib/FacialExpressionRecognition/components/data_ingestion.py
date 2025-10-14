import os
import zipfile
import gdown
from FacialExpressionRecognition import logger
from FacialExpressionRecognition.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the url
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded file from :[{dataset_url}] and saved at :[{zip_download_dir}]")
        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_dir = self.config.unzip_dir

        if not os.path.exists(unzip_dir):
            os.makedirs(unzip_dir, exist_ok=True)
            logger.info(f"Extracting zip file :[{self.config.local_data_file}] to dir :[{unzip_dir}]")
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)
            logger.info(f"Extracted zip file :[{self.config.local_data_file}] to dir :[{unzip_dir}]")
        else:
            logger.info(f"Zip file already extracted in dir :[{unzip_dir}]")