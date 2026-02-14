from sklearn.linear_model import LogisticRegression
import joblib

class ModelTrainer:
    def train(self,X,y):
      model = LogisticRegression(max_iter=1000)
      model.fit(X,y)
      joblib.dump(model, "artifacts/model.pkl")
      return model