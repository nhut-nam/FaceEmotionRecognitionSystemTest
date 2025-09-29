from FacialExpressionRecognition.config.configuration import ConfigurationManager 
from FacialExpressionRecognition.components.model_evalution_with_mlflow import ModelEvaluation
from FacialExpressionRecognition import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        model_eval = ModelEvaluation(config=eval_config)
        model_eval.evaluate()
        # model_eval.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise e