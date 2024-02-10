import numpy as np

from models import *
from helpers import *


def plot_graph_6(lambdas, train_accuracies, test_accuracies, validation_accuracies):
    plt.plot(lambdas, train_accuracies, label='Training Accuracy')
    plt.plot(lambdas, test_accuracies, label='Test Accuracy')
    plt.plot(lambdas, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('λ')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. λ')
    plt.legend()
    plt.show()


def plot_graph_7(vectors):
    # extract the x and y coordinates from the vectors
    x_coords = vectors[:, 0]
    y_coords = vectors[:, 1]

    # plot the points
    plt.scatter(x_coords, y_coords, color='blue', label='Vectors')

    # add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimized vector through time')

    plt.legend()
    plt.show()


def question_6(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    lambdas = [0, 2, 4, 6, 8, 10]

    models = np.empty(len(lambdas), dtype=object)

    train_accuracies = np.zeros(len(lambdas))
    validation_accuracies = np.zeros(len(lambdas))
    test_accuracies = np.zeros(len(lambdas))

    for i, lambd in enumerate(lambdas):
        ridge_regression_model = Ridge_Regression(lambd)
        ridge_regression_model.fit(X_train, Y_train)

        models[i] = ridge_regression_model
        # predict the labels of the sets
        y_preds_train = ridge_regression_model.predict(X_train)
        y_preds_test = ridge_regression_model.predict(X_test)
        y_preds_val = ridge_regression_model.predict(X_val)

        # compute the accuracies on every set
        train_accuracies[i] = np.mean(y_preds_train == Y_train)
        test_accuracies[i] = np.mean(y_preds_test == Y_test)
        validation_accuracies[i] = np.mean(y_preds_val == Y_val)

    plot_graph_6(lambdas, train_accuracies, test_accuracies, validation_accuracies)

    val_best_model_index = np.argmax(validation_accuracies)
    val_best_model_lambda = lambdas[val_best_model_index]
    print(f"Best model according to validation set is a model with lambda={val_best_model_lambda}",
          "The test accuracy of that model is: ", test_accuracies[val_best_model_index])
    plot_decision_boundaries(models[val_best_model_index], X_test, Y_test,
                             f"Best model (lambda= {val_best_model_lambda}) according to the validation set")

    val_worst_model_index = np.argmin(validation_accuracies)
    val_worst_model_lambda = lambdas[val_worst_model_index]
    plot_decision_boundaries(models[val_worst_model_index], X_test, Y_test,
                             f"Worst model (lambda= {val_worst_model_lambda}) according to the validation set")


def question_7():
    vectors = np.zeros([1000, 2])
    x = 0
    y = 0
    for i in range(1000):
        vectors[i, 0] = x
        vectors[i, 1] = y

        x -= 0.01 * (2 * (x - 3))  # the gradient of the function f according to x
        y -= 0.01 * (2 * (y - 5))  # the gradient of the function f according to y

    plot_graph_7(vectors)


if __name__ == '__main__':
    # read and load the data sets
    X_train, Y_train = read_data('train.csv')
    X_test, Y_test = read_data('test.csv')
    X_val, Y_val = read_data('validation.csv')

    # question_6(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    question_7()
