import numpy as np

from models import *
from helpers import *
from dataset import *


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

    magnitudes = np.linalg.norm(vectors, axis=1)
    normalized_magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
    alphas = 0.2 + 0.5 * normalized_magnitudes

    # plot the points
    plt.scatter(x_coords, y_coords, color='blue', label='Vectors', alpha=alphas)

    for i in range(len(vectors) - 1):
        plt.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color='blue', linestyle='--')

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

        x -= 0.1 * (2 * (x - 3))  # the gradient of the function f according to x
        y -= 0.1 * (2 * (y - 5))  # the gradient of the function f according to y

    plot_graph_7(vectors)


def check_gpu():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x = torch.arange(10)
    print("Tensor x:", x)
    print("Tensor x Device (CPU):", x.device)

    # Move the model and tensors to GPU if available
    x.to(device)

    print("Tensor x Device (GPU):", x.device)
    return device


def load_data(device):
    # create datasets
    train_dataset = Dataset(X_train, Y_train, device)
    test_dataset = Dataset(X_test, Y_test, device)
    val_dataset = Dataset(X_val, Y_val, device)

    # create data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,  # dataset to loop over
                                               batch_size=32,  # batch size
                                               shuffle=True  # if true shuffles the examples. Else uses
                                               )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader


def plot_lost_values(ep_loss_values):
    # Plot the loss values through epochs
    plt.plot(ep_loss_values, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Progression')
    plt.show()


def init_model(device, train_dataset):
    # Instantiate the model
    number_of_classes = len(torch.unique(train_dataset.labels))
    model = Logistic_Regression(X_train.shape[1], number_of_classes)
    model.to(device)
    return model


def train_model(device, model, optimizer, criterion, lr_scheduler, dataset, data_loader):
    loss_values = []
    ep_correct_preds = 0.
    model.train()  # set the model to training mode
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        # Store the loss values for plotting
        loss_values.append(loss.item())
        ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    lr_scheduler.step()

    mean_loss = np.mean(loss_values)
    ep_train_accuracy = ep_correct_preds / len(dataset)
    return mean_loss, ep_train_accuracy


def evaluate_model(device, model, dataset, data_loader):
    model.eval()  # set the model to evaluation mode
    ep_correct_preds = 0.
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, dim=1)
        ep_correct_preds += torch.sum(predicted == labels).item()

    ep_test_accuracy = ep_correct_preds / len(dataset)
    return ep_test_accuracy


def create_model(learning_rate):
    device = check_gpu()

    # Create datasets and data loaders
    train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader = load_data(device)

    # Instantiate the model,criterion, optimizer, lr_scheduler
    model = init_model(device, train_dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

    # Train the model for a few epochs with GPU acceleration
    num_epochs = 10
    train_loss_values = np.zeros(num_epochs)
    test_loss_values = np.zeros(num_epochs)
    val_loss_values = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)
    test_accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # Train the model
        train_loss_values[epoch], train_accuracies[epoch] = train_model(device, model, optimizer, criterion,
                                                                        lr_scheduler,
                                                                        train_dataset, train_loader)

        # Evaluate the model on the test set
        model.eval()
        test_loss_values[epoch], test_accuracies[epoch] = train_model(device, model, optimizer, criterion, lr_scheduler,
                                                                      test_dataset, test_loader)

        # Evaluate the model on the validation set
        val_loss_values[epoch], val_accuracies[epoch] = train_model(device, model, optimizer, criterion, lr_scheduler,
                                                                    val_dataset, val_loader)

        # print(
        #     f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss.item():.4f}, Train Accuracy: {ep_train_accuracy:.4f},'
        #     f' Test Accuracy: {ep_test_accuracy:.4f}, Validation Accuracy: {ep_val_accuracy:.4f}')
    model.set_accuracies(train_accuracies, test_accuracies, val_accuracies)
    return model


def question_9_3():
    model1 = create_model(0.1)
    model2 = create_model(0.01)
    model3 = create_model(0.001)
    print(
        f"Model with learning rate 0.1: Train Accuracy: {np.mean(model1.train_accuracies):.4f}, Test Accuracy: {np.mean(model1.test_accuracies):.4f}, "
        f"Validation Accuracy: {np.mean(model1.val_accuracies):.4f}")
    print(
        f"Model with learning rate 0.01: Train Accuracy: {np.mean(model2.train_accuracies):.4f}, Test Accuracy: {np.mean(model2.test_accuracies):.4f}, "
        f"Validation Accuracy: {np.mean(model2.val_accuracies):.4f}")
    print(
        f"Model with learning rate 0.001: Train Accuracy: {np.mean(model3.train_accuracies):.4f}, Test Accuracy: {np.mean(model3.test_accuracies):.4f},"
        f" Validation Accuracy: {np.mean(model3.val_accuracies):.4f}")

    plot_decision_boundaries(model3, X_test, Y_test, "Model with the best validation accuracy (learning rate= 0.01)")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # read and load the data sets
    X_train, Y_train = read_data('train.csv')
    X_test, Y_test = read_data('test.csv')
    X_val, Y_val = read_data('validation.csv')

    # question_6(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    # question_7()
    question_9_3()
