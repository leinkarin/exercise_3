import numpy as np
import sklearn.tree

from models import *
from helpers import *
from dataset import *
from plotters import *


def question_6():
    # read and load the data sets
    X_train, Y_train = read_data('train.csv')
    X_test, Y_test = read_data('test.csv')
    X_val, Y_val = read_data('validation.csv')

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

    plots_values = [(train_accuracies, 'Training Accuracy'), (test_accuracies, 'Test Accuracy'),
                    (validation_accuracies, 'Validation Accuracy')]
    plot_graph(plots_values, x_axis=lambdas, title='Accuracy vs. λ', x_label='λ', y_label='Accuracy')

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


def load_data(device, X_train, Y_train, X_test, Y_test, X_val, Y_val):
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


def init_model(device, train_dataset, X_train, learning_rate, step_size, gamma):
    # Instantiate the model
    number_of_classes = len(torch.unique(train_dataset.labels))
    model = Logistic_Regression(X_train.shape[1], number_of_classes, learning_rate)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return model, optimizer, lr_scheduler


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


def evaluate_model(device, model, criterion, dataset, data_loader):
    model.eval()  # set the model to evaluation mode
    loss_values = []
    ep_correct_preds = 0.
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # outputs = nn.functional.softmax(outputs, dim=1)
        loss = criterion(outputs.squeeze(), labels)
        # loss.backward()
        # _, predicted = torch.max(outputs, dim=1)
        loss_values.append(loss.item())
        ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    mean_loss = np.mean(loss_values)
    ep_accuracy = ep_correct_preds / len(dataset)
    return mean_loss, ep_accuracy


def create_model(X_train, Y_train, X_test, Y_test, X_val, Y_val, learning_rate, num_epochs, step_size=10, gamma=0):
    device = check_gpu()

    # Create datasets and data loaders
    train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader = load_data(device, X_train,
                                                                                                Y_train, X_test, Y_test,
                                                                                                X_val, Y_val)

    # Instantiate the model,criterion, optimizer, lr_scheduler
    model, optimizer, lr_scheduler = init_model(device, train_dataset, X_train, learning_rate, step_size, gamma)
    criterion = nn.CrossEntropyLoss()

    # define numpy arrays to keep all the values
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
        test_loss_values[epoch], test_accuracies[epoch] = evaluate_model(device, model, criterion,
                                                                         test_dataset, test_loader)

        # Evaluate the model on the validation set
        val_loss_values[epoch], val_accuracies[epoch] = evaluate_model(device, model, criterion,
                                                                       val_dataset, val_loader)

    print(
        f"model= {learning_rate}\n, train: {train_accuracies}\n, test: {test_accuracies}\n, validation:{val_accuracies}")
    model.set_accuracies(train_accuracies, test_accuracies, val_accuracies)
    model.set_losses(train_loss_values, test_loss_values, val_loss_values)
    return model


def find_best_model(models):
    best_model = models[0]
    for model in models:
        train_accuracy = np.mean(model.train_accuracies)
        test_accuracy = np.mean(model.test_accuracies)
        val_accuracy = np.mean(model.val_accuracies)
        print(
            f"Model with learning rate {best_model.learning_rate}: Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > np.mean(best_model.val_accuracies):
            best_model = model

    return best_model


def question_9_3():
    # read and load the data sets
    X_train, Y_train = read_data('train.csv')
    X_test, Y_test = read_data('test.csv')
    X_val, Y_val = read_data('validation.csv')

    num_epochs = 10
    learning_rates = [0.1, 0.01, 0.001]
    models = []

    # create three models with different learning rates
    for learning_rate in learning_rates:
        models.append(
            create_model(X_train, Y_train, X_test, Y_test, X_val, Y_val, learning_rate, num_epochs=num_epochs))

    best_model = find_best_model(models)

    plot_decision_boundaries(best_model, X_test, Y_test,
                             "Model with the best validation accuracy (learning rate= 0.001)")
    plots_values = [(best_model.train_losses, 'Train losses'), (best_model.test_losses, 'Test losses'),
                    (best_model.val_losses, 'Validation losses')]

    plot_graph(plots_values, x_axis=np.arange(num_epochs)
               , title='Loss values vs epochs', x_label='Epochs', y_label='Loss Value')


def question_9_4():
    # read and load the data sets
    X_train_mc, Y_train_mc = read_data('train_multiclass.csv')
    X_test_mc, Y_test_mc = read_data('test_multiclass.csv')
    X_val_mc, Y_val_mc = read_data('validation_multiclass.csv')

    num_epochs = 30
    learning_rates = [0.01, 0.001, 0.0003]
    models = []

    # create three models with different learning rates
    for learning_rate in learning_rates:
        models.append(
            create_model(X_train_mc, Y_train_mc, X_test_mc, Y_test_mc, X_val_mc, Y_val_mc, learning_rate,
                         num_epochs=num_epochs, step_size=5, gamma=0.3))

    # create a list of the mean test and validation accuracies of every model
    last_test_accuracies = []
    last_val_accuracies = []
    for model in models:
        last_test_accuracies.append(model.test_accuracies[num_epochs - 1])
        last_val_accuracies.append(model.val_accuracies[num_epochs - 1])

    # plot a graph learning rate vs accuracies
    plot_values = [(last_test_accuracies, 'Test accuracy'), (last_val_accuracies, 'Validation accuracy')]
    plot_graph(plot_values, learning_rates,
               title="Learning rate vs accuracies", x_label='Learning rate', y_label='Accuracy')

    # find best model
    best_model = find_best_model(models)
    print(
        f"The test accuracy of the best model (learning rate={best_model.learning_rate}) "
        f"according to the validation accuracy is {best_model.test_accuracies[num_epochs - 1]}")

    loss_values = [(best_model.train_losses, 'Train losses'), (best_model.test_losses, 'Test losses'),
                   (best_model.val_losses, 'Validation losses')]

    plot_graph(loss_values, x_axis=np.arange(num_epochs)
               , title='Loss values vs epochs', x_label='Epochs', y_label='Loss Value')

    accuracies = [(best_model.train_accuracies, 'Train accuracy'), (best_model.test_accuracies, 'Test accuracy'),
                  (best_model.val_accuracies, 'Validation accuracy')]
    plot_graph(accuracies, x_axis=np.arange(num_epochs),
               title='Accuracies vs echos', x_label='Epochs', y_label='Accuracies')


def trees(max_depth):
    # read and load the data sets
    X_train_mc, Y_train_mc = read_data('train_multiclass.csv')
    X_test_mc, Y_test_mc = read_data('test_multiclass.csv')
    X_val_mc, Y_val_mc = read_data('validation_multiclass.csv')

    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)

    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train_mc, Y_train_mc)

    # make predictions
    Y_pred_train = tree_classifier.predict(X_train_mc)
    Y_pred_test = tree_classifier.predict(X_test_mc)
    Y_pred_val = tree_classifier.predict(X_val_mc)

    # compute the accuracies of the predictions
    accuracy_train = np.mean(Y_pred_train == Y_train_mc)
    accuracy_test = np.mean(Y_pred_test == Y_test_mc)
    accuracy_val = np.mean(Y_pred_val == Y_val_mc)

    plot_decision_boundaries(tree_classifier, X_val_mc, Y_val_mc, f'Decision tree with max depth {max_depth}')
    print(
        f"The train accuracy of the model is: {accuracy_train},The test accuracy of the model is: {accuracy_test},The validation accuracy of the model is: {accuracy_val}")


def bonus():
    # read and load the data sets
    X_train_mc, Y_train_mc = read_data('train_multiclass.csv')
    X_test_mc, Y_test_mc = read_data('test_multiclass.csv')
    X_val_mc, Y_val_mc = read_data('validation_multiclass.csv')

    num_epochs = 30
    learning_rate = 0.01
    lambdas = [0, 2, 4, 6, 8, 10]
    best_model = None
    best_val_accuracy = 0

    for lambd in lambdas:
        model = create_model_with_ridge(X_train_mc, Y_train_mc, X_test_mc, Y_test_mc, X_val_mc, Y_val_mc,
                                        learning_rate, num_epochs=num_epochs, lambda_val=lambd)
        val_accuracy = model.val_accuracies[-1]
        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy

    print(f"The test accuracy of the best model (lambda={best_model.lambda_val}) "
          f"according to the validation accuracy is {best_model.test_accuracies[-1]}")

    plot_decision_boundaries(best_model, X_val_mc, Y_val_mc, f'Desicion boundries of the best model (λ=0)')


def create_model_with_ridge(X_train, Y_train, X_test, Y_test, X_val, Y_val, learning_rate, num_epochs, lambda_val):
    device = check_gpu()

    # Create datasets and data loaders
    train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader = load_data(device, X_train,
                                                                                                Y_train, X_test, Y_test,
                                                                                                X_val, Y_val)

    # Instantiate the model, criterion, optimizer, lr_scheduler
    model, optimizer, lr_scheduler = init_model(device, train_dataset, X_train, learning_rate, 10, 0.3)
    criterion = nn.CrossEntropyLoss()

    # define numpy arrays to keep all the values
    train_loss_values = np.zeros(num_epochs)
    test_loss_values = np.zeros(num_epochs)
    val_loss_values = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)
    test_accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # Train the model
        train_loss_values[epoch], train_accuracies[epoch] = train_model_with_ridge(device, model, optimizer, criterion,
                                                                                   lr_scheduler,
                                                                                   train_dataset, train_loader,
                                                                                   lambda_val)

        # Evaluate the model on the test set
        test_loss_values[epoch], test_accuracies[epoch] = evaluate_model(device, model, criterion,
                                                                         test_dataset, test_loader)

        # Evaluate the model on the validation set
        val_loss_values[epoch], val_accuracies[epoch] = evaluate_model(device, model, criterion,
                                                                       val_dataset, val_loader)

    print(f"Model with lambda={lambda_val}: Train Accuracy: {np.mean(train_accuracies):.4f}, "
          f"Test Accuracy: {np.mean(test_accuracies):.4f}, Validation Accuracy: {np.mean(val_accuracies):.4f}")

    model.set_accuracies(train_accuracies, test_accuracies, val_accuracies)
    model.set_losses(train_loss_values, test_loss_values, val_loss_values)
    model.lambda_val = lambda_val

    return model


def train_model_with_ridge(device, model, optimizer, criterion, lr_scheduler, dataset, data_loader, lambda_val):
    loss_values = []
    ep_correct_preds = 0.
    model.train()  # set the model to training mode
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        # Add ridge regularization to the loss
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param) ** 2
        loss += lambda_val * l2_reg

        loss.backward()
        optimizer.step()

        # Store the loss values for plotting
        loss_values.append(loss.item())
        ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    lr_scheduler.step()

    mean_loss = np.mean(loss_values)
    ep_train_accuracy = ep_correct_preds / len(dataset)
    return mean_loss, ep_train_accuracy


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    question_6()
    question_7()
    question_9_3()
    question_9_4()
    trees(2)
    trees(10)
    bonus()
