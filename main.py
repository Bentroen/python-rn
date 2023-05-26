from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

inputs = [
    "01001100010001001110",  # 1
    "01101001001001001111",  # 2
    "11100001001000011110",  # 3
    "10101010111100100010",  # 4
    "11111000111000011110",  # 5
    "01111000111010010110",  # 6
    "11110001001001000100",  # 7
    "01101001011010010110",  # 8
    "01101001011100011111",  # 9
    "01101001100110010110",  # 0
]


outputs = [
    "1000000000",  # 1
    "0100000000",  # 2
    "0010000000",  # 3
    "0001000000",  # 4
    "0000100000",  # 5
    "0000010000",  # 6
    "0000001000",  # 7
    "0000000100",  # 8
    "0000000010",  # 9
    "0000000001",  # 0
]


class NeuralNetwork(nn.Module):
    def __init__(self, nnets: int):
        super(NeuralNetwork, self).__init__()
        # first layer nnets neurons and tanh activation
        self.fc1 = nn.Linear(20, nnets)
        self.fc1.activation = nn.Tanh()
        # second layer 10 neurons and linear activation and softmax activation 0-1
        self.fc2 = nn.Linear(nnets, 10)

        self.model = nn.Sequential(self.fc1, self.fc2)

    def forward(self, x):
        x = self.model(x)
        return x


def read_tiles_from_image(image_path: str) -> List[List[int]]:
    # open the image "numeros.png"
    img = Image.open(image_path)
    # the image is a 40 x 5 image
    # split the image into 10 4 x 5 images

    # convert the image to black and white
    img = img.convert("1")

    numbers = []
    for i in range(10):
        # crop the image
        numbers.append(img.crop((i * 4, 0, (i + 1) * 4, 5)))

    # create a list of bit strips of the numbers
    # the index of the list is the number
    numbers_list = []
    for i in range(10):
        numbers_list.append([])
        for j in range(5):
            for k in range(4):
                numbers_list[i].append(0 if numbers[i].getpixel((k, j)) == 255 else 1)

    print("Pixels of the numbers in the image:")
    for i in numbers_list:
        print("".join(str(x) for x in i))

    return numbers_list


def train_network(
    input_tensors: torch.Tensor,
    output_tensors: torch.Tensor,
    nnets: int,
    epochs: int,
    goal: float,
    learning_rate: float,
    momentum: float,
) -> NeuralNetwork:
    # Mean Squared Loss
    criterion = nn.MSELoss()
    model = NeuralNetwork(nnets)
    # stochastic gradient descent
    optimizer = optim.SGD(  # Stochastic Gradient Descent/Gradient Descent with Momentum (TRAINGDM Equivalent)
        model.parameters(), lr=learning_rate, momentum=momentum
    )

    # Train the neural network
    for epoch in range(epochs):
        # Forward pass
        outputs = model(input_tensors)
        loss = criterion(outputs, output_tensors)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}, Learning Rate: {learning_rate}, Momentum: {momentum}"
            )

        if loss.item() < goal:
            break

    print(f"Final loss: {loss.item():.6f}")
    return model


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    softmax = nn.Softmax(dim=1)
    normalized_tensor = softmax(tensor)
    return normalized_tensor


def make_flip_n_bits(bits: torch.Tensor, n: int = 1) -> torch.Tensor:
    import random

    randomIdxs = random.sample(range(0, len(bits)), n)
    for i in randomIdxs:
        bits[i] = 1 if bits[i] == 0 else 0
    return bits


def test_network(
    model: NeuralNetwork, input_tensors: torch.Tensor, output_tensors: torch.Tensor
):
    original_input_tensors = input_tensors.clone()

    # Test the neural network
    print("Testing the neural network")
    with torch.no_grad():
        outputs = model(input_tensors)
        print(f"Predicted: {normalize_tensor(outputs)}")
        print(f"Actual: {output_tensors}")

    input_tensors = original_input_tensors.clone()

    # flip 1 bit in each input and test again
    print("Testing the neural network with 1 bit flipped in each input")
    for i in range(len(input_tensors)):
        input_tensors[i] = make_flip_n_bits(input_tensors[i], 1)
    with torch.no_grad():
        outputs = model(input_tensors)
        print(f"Predicted: {normalize_tensor(outputs)}")
        print(f"Actual: {output_tensors}")

    input_tensors = original_input_tensors.clone()
    input("Press enter to continue...")
    # flip 2 bits in each input and test again
    print("Testing the neural network with 2 bits flipped in each input")
    for i in range(len(input_tensors)):
        input_tensors[i] = make_flip_n_bits(input_tensors[i], 2)
    with torch.no_grad():
        outputs = model(input_tensors)
        print(f"Predicted: {normalize_tensor(outputs)}")
        print(f"Actual: {output_tensors}")

    input_tensors = original_input_tensors.clone()

    # flip 3 bits in each input and test again
    print("Testing the neural network with 3 bits flipped in each input")
    for i in range(len(input_tensors)):
        input_tensors[i] = make_flip_n_bits(input_tensors[i], 3)
    with torch.no_grad():
        outputs = model(input_tensors)
        print(f"Predicted: {normalize_tensor(outputs)}")
        print(f"Actual: {output_tensors}")


def tests(input_tensors: torch.Tensor, output_tensors: torch.Tensor):
    train_params = [
        {"epochs": 10000, "goal": 0.0005, "learning_rate": 0.1, "momentum": 0.0},
        {"epochs": 10000, "goal": 0.0005, "learning_rate": 0.4, "momentum": 0.0},
        {"epochs": 10000, "goal": 0.0005, "learning_rate": 0.9, "momentum": 0.0},
        {"epochs": 10000, "goal": 0.0005, "learning_rate": 0.1, "momentum": 0.4},
        {"epochs": 10000, "goal": 0.0005, "learning_rate": 0.9, "momentum": 0.4},
    ]
    neural_networks = [15, 25, 35]
    for nnets in neural_networks:
        for params in train_params:
            epochs = params["epochs"]
            goal = params["goal"]
            learning_rate = params["learning_rate"]
            momentum = params["momentum"]

            trained_model = train_network(
                input_tensors,
                output_tensors,
                nnets,
                epochs,
                goal,
                learning_rate,
                momentum,
            )
            test_network(trained_model, input_tensors, output_tensors)


def main():
    data = read_tiles_from_image("numeros.png")

    input_tensors = torch.tensor(data, dtype=torch.float)
    output_tensors = torch.tensor(
        [list(map(int, outp)) for outp in outputs], dtype=torch.float
    )
    tests(input_tensors, output_tensors)


if __name__ == "__main__":
    main()
