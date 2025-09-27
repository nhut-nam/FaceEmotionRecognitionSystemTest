from FacialExpressionRecognition.config.configuration import ConfigurationManager 
from FacialExpressionRecognition.components.prepare_base_model import PrepareBaseModel
from FacialExpressionRecognition import logger

STAGE_NAME = "Prepare Base Model stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(prepare_base_model_config)
        prepare_base_model.update_base_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<")
        logger.exception(e)