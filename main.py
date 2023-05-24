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


def read_tiles_from_image(image_path):
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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 10)  # 10 inputs, 10 outputs

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


def main():
    data = read_tiles_from_image("numeros.png")

    input_tensors = torch.tensor(data, dtype=torch.float)
    output_tensors = torch.tensor(
        [list(map(int, outp)) for outp in outputs], dtype=torch.float
    )

    # Initialize the neural network
    model = NeuralNetwork()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Train the neural network
    for epoch in range(1000):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(input_tensors)  # Forward pass
        loss = criterion(outputs, output_tensors)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        if epoch % 100 == 0:
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

    # Test the model on a sample input
    sample_input = torch.tensor(
        [list(map(int, "01101001001001001111"))], dtype=torch.float
    )
    prediction = model(sample_input)
    predicted_digit = prediction.argmax().item()
    print(f"Predicted digit: {predicted_digit}")


if __name__ == "__main__":
    main()
