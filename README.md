# CIFAR-10 Image Classification using CNN (PyTorch)

## Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to perform image classification on the CIFAR-10 dataset. The objective is to classify 32x32 RGB images into one of 10 categories using deep learning.

The final trained model achieves a **test accuracy of approximately 75.85%**.

## Dataset
The project uses the CIFAR-10 dataset, which contains:

- 60,000 color images (32x32 pixels)
- 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- 50,000 training images
- 10,000 test images

Note: The dataset is automatically downloaded using `download=True` in torchvision when running the script.


## Model Architecture
The CNN model includes:

- Convolutional layers
- ReLU activation functions
- Max pooling layers
- Fully connected (Dense) layers
- Softmax output layer

The model is trained using categorical cross-entropy loss and optimized using backpropagation.


## Training & Validation
- Training loss and validation loss were calculated during training.
- A graph of training loss vs validation loss was plotted to monitor model performance.
- Validation loss initially decreased and later increased after certain epochs, indicating overfitting.
- The best-performing model (lowest validation loss) was saved and used for final test evaluation.


## Results
- Test Accuracy: **~75.85%**
- Loss curves were used to analyze convergence and generalization behavior.


## Requirements
- Python 3.x
- torch
- torchvision
- numpy
- matplotlib

You can install the required libraries using:

pip install torch torchvision numpy matplotlib


## How to Run

1. Clone the repository:

   git clone <repository-url>

2. Navigate to the project folder:

   cd <project-folder-name>

3. Run the training script:

   python train.py

The CIFAR-10 dataset will be downloaded automatically if not already present.


## Conclusion
This project demonstrates the practical implementation of a CNN for multi-class image classification using CIFAR-10. It highlights the importance of monitoring validation loss to detect overfitting and selecting the best-performing model for evaluation.
