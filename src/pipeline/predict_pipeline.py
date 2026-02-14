import joblib 
import pandas as pd 

class PredictPipeline:
    def predict(self, data:pd.DataFrame):

        model= joblib.load("artifacts/model.pkl")
        preprocessor = joblib.load("artifacts/preprocessor.pkl")
        feature_names = joblib.load("artifacts/feature_names.pkl")
        
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])


        for col in feature_names:
            if col not in data.columns:
                raise ValueError(f"Missing required feature:{col}")
            
        df = df[feature_names]

        X = preprocessor.transform(df)
        return model.predict(X)