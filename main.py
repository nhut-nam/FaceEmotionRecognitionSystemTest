from FacialExpressionRecognition import logger
from FacialExpressionRecognition.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from FacialExpressionRecognition.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from FacialExpressionRecognition.pipeline.stage_03_training_model import TrainingPipeline
from FacialExpressionRecognition.pipeline.stage_04_model_evalution_with_mlflow import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed {e} <<<<<<")



STAGE_NAME = "Prepare Base Model stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<")
    logger.exception(e)


STAGE_NAME = "Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<")
    logger.exception(e)

STAGE_NAME = "Model Evaluation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<")
    logger.exception(e)