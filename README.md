# CNN vs ResNet50: Drone Classification

This project presents a comparative study between two deep learning models—Custom Convolutional Neural Networks (CNN) and Fine-tuned ResNet50—for the task of drone image classification. The project includes the code implementation, model training, evaluation, and analysis using Python and deep learning frameworks.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Conclusion](#conclusion)
- [Files](#files)

## Project Overview

This project compares the performance of a Custom CNN and a Fine-tuned ResNet50 model for drone classification (drone vs. non-drone images). The study evaluates and contrasts the models based on training time, accuracy, computational complexity, and performance on test data.

## Installation

To get started, clone the repository and install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/akshaysatyam2/CNN-vs-ResNet50-Classification.git
   ```

## Dataset

Data was obtained from web through freely available websites, such as:
The dataset was obtained from several publicly available sources, including:

- [Unsplash](https://unsplash.com/)
- [Pixelied](https://pixelied.com/)
- [iStock](https://www.istockphoto.com/)

The complete dataset is available on Kaggle: [Drone and Non-Drone Dataset](https://www.kaggle.com/datasets/akshaysatyam2/drone-and-not-drone)


And API command to get is: 
```cmd
kaggle datasets download -d akshaysatyam2/drone-and-not-drone
```

## Model Architectures

1)	ResNet Model: The model architecture consists of a pre-trained ResNet50 base with added layers for classification. The final layer employs a sigmoid activation function for binary classification.
2)	Model from Scratch: The model architecture consists of a simple CNN with two convolutional layers followed by max-pooling layers for feature extraction. The flattened output is fed into two fully connected (dense) layers for classification, with a sigmoid activation function in the output layer for binary classification.

## Results

Both models were trained for 25 epochs, and their performance was evaluated on a test dataset. Below are the key results:

Custom CNN
- Test Accuracy: 90.91%
- Training Time: Faster due to fewer layers and lower complexity.
- Generalization: Satisfactory, but struggles on complex tasks.

Fine-tuned ResNet50
- Test Accuracy: 93.60%
- Training Time: Slower due to the larger model and pre-trained layers.
- Generalization: Excellent, benefiting from transfer learning.

## Conclusion

- The Custom CNN is quick to implement and train, making it suitable for small-scale projects with limited resources.

- The Fine-tuned ResNet50 offers superior performance and accuracy, making it ideal for more complex tasks or larger datasets. It benefits from pre-trained weights on large datasets like ImageNet, providing excellent generalization.

## Files

- [drones-classification.ipynb](https://github.com/akshaysatyam2/CNN-vs-ResNet50-Classification/blob/master/drones-classification.ipynb): Jupyter notebook for training and evaluating both CNN and ResNet50 models.
- [Comparative Study of CNN vs. Fine-tuned ResNet50.docx](https://github.com/akshaysatyam2/CNN-vs-ResNet50-Classification/blob/master/Comparative%20Study%20of%20CNN%20vs.%20Fine-tuned%20ResNet50.docx): Detailed documentation of the project.
- [Comparative Study of CNN vs. Fine-tuned ResNet50.pdf](https://github.com/akshaysatyam2/CNN-vs-ResNet50-Classification/blob/master/Comparative%20Study%20of%20CNN%20vs.%20Fine-tuned%20ResNet50.pdf): PDF version of the documentation.
- [cnn_history.png](https://github.com/akshaysatyam2/CNN-vs-ResNet50-Classification/blob/master/cnn_history.png): Plot showing the training loss and accuracy of the CNN model.
- [resNet_history.png](https://github.com/akshaysatyam2/CNN-vs-ResNet50-Classification/blob/master/resNet_history.png): Plot showing the training loss and accuracy of the ResNet50 model.
