from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from dataset import Iris

# Extreme Gradient Boosting (e.g., XGBoost, LightGBM): 
# Ensemble learning methods that build a strong predictive model by combining the predictions of multiple weak models sequentially.
class XGBBoost():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.xgb_boost = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

    def train(self, X_test, y_test):
        return self.xgb_boost.fit(X_test, y_test)

    def predict(self, X_test):
        return self.xgb_boost.predict(X_test)

if __name__ == '__main__':
    random_seed=1
    n_clusters=3
    n_init=10

    iris = Iris()
    ada = XGBBoost()

    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)

    # test
    ada.train(X_train, y_train)
    y_pred = ada.predict(X_test)

    accuracy_ = accuracy_score(y_test, y_pred)
    print(f"XGBBoost Accuracy scratch: {accuracy_:.4f}")
