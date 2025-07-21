import argparse
import json
import sys
import torch
import torch.nn as nn
import torch.utils.data as td


def main(args):

    # Read configuration file
    with open("../config/user_config_file.json") as f:
        config_file = json.load(f)

    # Import helper functions from src/utils_MNIST.py
    sys.path.append(config_file["utils_location"])

    # Fix seed
    torch.manual_seed(42)

    # Inform user about input parameters
    if args.DEBUG:
        print("\nTRAINING SPECIFICATION:")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of training epochs: {args.epochs}")

    # Data loaders for train, validation and test data
    train_loader, valid_loader, test_loader = create_MNIST_data_loaders(args.batch_size)

    # Get the neural network specification
    neural_network_specification = design_neural_network(args.DEBUG)

    # Train the neural network
    loss_criteria = nn.CrossEntropyLoss()
    neural_network_specification = design_neural_network(args.DEBUG)
    optimizer_adam = torch.optim.Adam(neural_network_specification.parameters(), lr = args.learning_rate)
    train_loss_list, validation_loss_list, validation_accuracy_list = [], [], []
    print("\n--------------------------- START TRAINING ---------------------------")
    for epoch in range(args.epochs):

        # Train model
        train_loss = train_model(neural_network_specification, train_loader, optimizer_adam)
        validation_loss, validation_correct = evaluate_model(neural_network_specification, valid_loader)

        # Inform user
        if args.DEBUG:
            print(f"EPOCH {epoch+1} OUT OF {args.epochs}")
            print(f"\t Training loss: {train_loss}")
            print(f"\t Validation loss: {validation_loss} -- Validation accuracy: {validation_correct}%")

        # Store stats
        train_loss_list.append(train_loss), validation_loss_list.append(validation_loss), validation_accuracy_list.append(validation_correct)
    print("\n--------------------------- END TRAINING ---------------------------")

    # Performance on the test data
    test_loss, test_correct = evaluate_model(neural_network_specification, test_loader)
    print(f"\nTest accuracy: {test_correct}%")
    

def create_MNIST_data_loaders(BATCH_SIZE: int):


    # Import utils_MNIST library
    import utils_MNIST

    # Read data from folder
    X_train, X_valid, X_test, y_train_digits, y_valid_digits, y_test_digits = utils_MNIST.prepare_MNIST_data_sets("../data/MNIST/")

    # Create a dataset and loader for TRAINING
    train_x = torch.Tensor(X_train).float()
    train_y = torch.Tensor(y_train_digits.copy()).long()
    train_ds = td.TensorDataset(train_x, train_y)
    train_loader = td.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # Create a dataset and loader for VALIDATION
    valid_x = torch.Tensor(X_valid).float()
    valid_y = torch.Tensor(y_valid_digits.copy()).long()
    valid_ds = td.TensorDataset(valid_x, valid_y)
    valid_loader = td.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # Create a dataset and loader for TESTING
    test_x = torch.Tensor(X_test).float()
    test_y = torch.Tensor(y_test_digits.copy()).long()
    test_ds = td.TensorDataset(test_x, test_y)
    test_loader = td.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    return train_loader, valid_loader, test_loader


def design_neural_network(DEBUG: bool):

    # Layer specification (input is 28x28=784, hidden layer with 800 units, output maps to 10 digits)
    layer_units = [784, 800, 10]

    # Define fully connected neural network
    class MNIST_fc(nn.Module):
        def __init__(self):
            super(MNIST_fc, self).__init__()
            self.fc1 = nn.Linear(layer_units[0], layer_units[1])
            self.fc2 = nn.Linear(layer_units[1], layer_units[2])
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return x
    
    # Instantiate neural network
    fullyconnected_nn = MNIST_fc()
    if DEBUG:
        total_params = [torch.numel(paraset) for paraset in fullyconnected_nn.parameters()]
        print("\nNEURAL NETWORK LAYOUT")
        print(f"Size of parameter sets: {total_params}")
        print(f"Total number of parameters: {sum(total_params)}")

    return fullyconnected_nn

def train_model(model, data_loader, optimizer):

    # Training mode
    model.train()

    # Define loss
    loss_criteria = nn.CrossEntropyLoss()

    
    # Update parameters based on datasets of BATCH_SIZE images
    train_loss = 0
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_criteria(prediction, target)
        train_loss += loss.item()

        # Backward propagation
        loss.backward()
        optimizer.step()

    return train_loss/(batch+1)


def evaluate_model(model, data_loader):

    # Evaluation mode
    model.eval()

    # Define loss
    loss_criteria = nn.CrossEntropyLoss()

    # Evaluate performance
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor
            prediction = model(data)
            loss = loss_criteria(prediction, target)
            total_loss += loss.item()

            # Class prediction
            predicted_class = torch.argmax(prediction, 1)
            correct_predictions += torch.sum(predicted_class==target).item()
        
        return total_loss/(batch+1), (100*correct_predictions/len(data_loader.dataset))


def parse_args():
    # Instantiate parser
    parser = argparse.ArgumentParser()

    # Parse arguments
    parser.add_argument("--learning_rate", dest="learning_rate", type=float)
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--epochs", dest="epochs", type=int)
    parser.add_argument("--DEBUG", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Run main function
    main(args)