# Cats and Dogs Classification with CNN

## Overview
This project aims to classify images of cats and dogs using convolutional neural networks (CNNs). The CNN model is trained to accurately identify whether an input image contains a cat or a dog.

## Features
- Utilizes PyTorch framework for CNN implementation.
- Provides a simple and effective solution for classifying cat and dog images.
- Offers a straightforward approach for training and testing the CNN model.

## Usage
1. **Dataset**: Prepare a dataset containing images of cats and dogs. Ensure that the dataset is properly organized with separate folders for each class.
2. **Training**: Train the CNN model using the provided training script. Adjust hyperparameters as needed.
3. **Evaluation**: Evaluate the trained model on a separate validation set to assess its performance.
4. **Inference**: Use the trained model to make predictions on new images of cats and dogs.

## Requirements
- Python 3.x
- PyTorch
- Other dependencies as specified in the project documentation

## Contributions
Contributions are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.


## Shape Inference
To determine the shape of input for fully connected layers, random input values are passed through initial layers, and the shape of their output is observed. For example:

## Architecture
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels =32, kernel_size =5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)
        self.fc1 = nn.Linear( 9 * 9 * 64, 256)
        self.fc2 = nn.Linear(256, 2) #since 2 output class 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 2)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1) # Last layer. So act = softmax, since binary classify dim =1
        return x
For detailed code and usage instructions, please refer to the provided scripts and documentation.



