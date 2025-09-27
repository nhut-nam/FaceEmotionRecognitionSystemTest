from FacialExpressionRecognition import logger
from FacialExpressionRecognition.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from FacialExpressionRecognition.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

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