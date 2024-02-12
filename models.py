import numpy as np
import torch
from torch import nn


class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.ridge_weights = None

    def fit(self, X, Y):
        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5)  # transform the labels to -1 and 1, instead of 0 and 1.
        N = X.shape[0]
        identity_matrix = np.eye(X.shape[1])
        self.ridge_weights = np.linalg.inv((X.T @ X) / N + self.lambd * identity_matrix) @ ((X.T @ Y) / N)

        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

        ####################################

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        result = X @ self.ridge_weights
        preds = np.where(result >= 0, 1, 0)  # 1 if the result is positive and 0 else

        # compute the predicted output of the model.
        # name your predicitons array preds.

        ####################################

        # transform the labels to 0s and 1s, instead of -1s and 1s.
        # You may remove this line if your code already outputs 0s and 1s.
        # preds = (preds + 1) / 2

        return preds


class Logistic_Regression(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.train_accuracies = None
        self.test_accuracies = None
        self.val_accuracies = None
        self.train_losses = None
        self.test_losses = None
        self.val_losses = None
        """
        :param input_dim: the number of features (2- long, lat)
        :param output_dim: the number of classes (2- 0,1)
        """
        super(Logistic_Regression, self).__init__()

        # define a linear operation.
        self.linear = nn.Linear(input_dim, output_dim)
        ####################################

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        output = self.linear(x)
        return output

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x

    def set_accuracies(self, train_accuracies, test_accuracies, val_accuracies):
        self.train_accuracies = train_accuracies
        self.test_accuracies = test_accuracies
        self.val_accuracies = val_accuracies

    def set_losses(self, train_losses, test_losses, val_losses):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.val_losses = val_losses
