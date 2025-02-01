import seaborn as sns
import matplotlib.pyplot as plt

from dataset import Iris

class Stats:
    def __init__(self):
        self.iris = Iris()

    def sns_pairplot_defalut(self, show=True):
        df = self.iris.panda_dataframe()
        sns.pairplot(df, hue='Species', palette=self.iris.fixed_palette, hue_order=self.iris.species_order, height=2)
        plt.get_current_fig_manager().set_window_title("Default")
        if show:
            plt.show()

    def sns_pairplot(self, X, y, show_default=False):
        df = self.iris.to_panda_dataframe(X, y)
        sns.pairplot(df, hue='Species', palette=self.iris.fixed_palette, hue_order=self.iris.species_order, height=2)
        plt.get_current_fig_manager().set_window_title("X and y")
        if show_default:
            self.sns_pairplot_defalut(show=False)
        plt.show()

    def scatterplot(self, X, y, show_default=False):
        df = self.iris.to_panda_dataframe(X, y)
        sns.scatterplot(df, hue='Species', height=2)
        plt.show()

if __name__ == '__main__':
    iris = Iris()
    stats = Stats()
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2)
    stats.sns_pairplot(X_train, y_train, show_default=True)
