import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(1)

class Iris:
    species_order = ['setosa', 'versicolor', 'virginica']
    fixed_palette = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}

    def __init__(self):
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.iris = load_iris()

    def normilize(self, X, fit=False):
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def denormilize(self, X):
        return self.scaler.inverse_transform(X)

    def to_panda_dataframe(self, X=None, y=None):
        import pandas as pd
        df = pd.DataFrame(X, columns=self.iris.feature_names)
        y = np.argmax(y, axis=1)
        df['Species'] = self.iris.target_names[y]
        return df

    def array_binary(self, X):
        return np.eye(X.shape[1])[np.argmax(X, axis=1)]

    def class_names(self):
        return self.iris.target_names

    def feature_names(self):
        return self.iris.feature_names

    def panda_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        df['Species'] = self.iris.target_names[self.iris.target]
        return df

    def array_dataset(self, one_hot_y=True, normilize=False, test_size=0.0):
        X = self.iris.data
        y = self.iris.target

        if one_hot_y:
            N = np.max(self.iris.target) + 1
            y = np.eye(N)[y]
        if normilize:
            X = self.normilize(X, fit=True)
        if test_size > 0.0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
            return X_train, X_test, y_train, y_test
        else:
            return X, y

    def numpy_dataset(self, test_size=0.2, one_hot_y=True, normilize=False):
        X, y = self.array_dataset(one_hot_y=one_hot_y, normilize=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
        if normilize:
            X_train = self.normilize(X_train, fit=True)
            X_test = self.normilize(X_test, fit=False)
        return X_train, X_test, y_train, y_test

    def torch_dataset(self, batch_size=16, normilize=False):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        if normilize:
            data = self.normilize(self.iris.data, fit=True)
        tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def show_probability(self, x, y):
        norm = np.zeros_like(y)
        norm[np.arange(y.shape[0]), np.argmax(y, axis=1)] = 1
        for a, b in zip(x, norm):
            print(a, b)

    def show_comparision(self, pred, y_test=None):
        equal = 0
        norm = np.zeros_like(pred)
        norm[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1
        y_test = y_test if y_test is not None else np.zeros_like(pred)
        for a, b in zip(y_test, norm):
            equal += (a == b).all()
            print(a, b)
        print(f"Accuracy = {int(100*equal/len(y_test))}%")

if __name__ == '__main__':
    iris = Iris()
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.01)
    for a, b in zip(X_train, y_train):
        print(a, b)