from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator

def run_training():
    ingestion = DataIngestion("config/config.yaml")
    df = ingestion.ingest()

    DataValidation().validate(df)

    X,y = DataTransformation().transform(df,"churn")

    model = ModelTrainer().train(X,y)

    ModelEvaluator().evaluate(model,X,y)

if __name__ == "__Main__":
    run_training()