# CIFAR-10 Dataset Deep Learning Project

CIFAR-10 is a widely used dataset in the field of computer vision. This project focuses on training and testing a deep learning model on the CIFAR-10 dataset. The objective is to develop a classification model and evaluate its performance.

## Dataset Analysis

The CIFAR-10 dataset has the following characteristics:
- **Dimensions**: Each image is 32x32 pixels with 3 channels (RGB).
- **Classes**: A total of 10 classes (e.g., airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).
- **Distribution**: 50,000 images for training, 10,000 images for testing.

### Data Preparation Process

1. **Data Normalization**: Image pixel values are scaled to the [0,1] range to improve model efficiency.
2. **One-Hot Encoding of Labels**: Labels are converted into vectors to facilitate model understanding.
3. **Training and Validation Sets**: The training set is split into 80% training and 20% validation.

## Model Definition

This project uses a Convolutional Neural Network (CNN) model with the following structure:

1. **Convolutional Layers**: Extract features from images.
2. **MaxPooling Layers**: Summarize feature maps.
3. **Dropout Layers**: Reduce the risk of overfitting.
4. **Dense Layers**: Fully connected layers for classification.
5. **Activation Functions**: ReLU (for hidden layers) and Softmax (for the output layer).

The model is compiled with the `adam` optimizer and `categorical_crossentropy` loss function.

## Model Training

The model was trained for a total of 20 epochs using the training dataset. During training, accuracy and loss values for both the training and validation sets were recorded.

### Training Results

- **Training Accuracy**: The model’s accuracy on the training set steadily improved.
- **Validation Accuracy**: Validation accuracy generally increased but showed occasional fluctuations.

## Performance on the Test Set

- **Test Accuracy**: The model achieved approximately **71.23% accuracy** on the test set.
- **Test Loss**: The model achieved an average **0.8401 loss** on the test set.

### Confusion Matrix

The confusion matrix visualizes which classes were correctly or incorrectly classified.

## Data Augmentation

To improve model generalization and performance, data augmentation techniques are applied. This generates new variations of the input images to simulate a larger and more diverse dataset.

### Model Training with Data Augmentation

The model was retrained for 20 epochs using data augmentation.

### Performance Evaluation

- **Metrics**:
  - **Test Accuracy**: `74.35%`
  - **Test Loss**: `0.7520`
- **Confusion Matrix**: Reduced misclassifications, especially for similar classes.

## Results and Comparisons

### Key Observations

1. **Improved Generalization**:
   - The test accuracy increased from `71.23%` to `74.35%`, demonstrating that data augmentation helped the model generalize better to unseen data.
   - Test loss decreased from `0.8401` to `0.7520`, further supporting improved performance.

2. **Validation Trends**:
   - With augmentation, the validation accuracy curve shows more stability over epochs compared to training without augmentation.
   - The gap between training and validation accuracy is narrower, suggesting reduced overfitting.

3. **Class-Specific Improvements**:
   - The confusion matrix highlights reduced errors for visually similar classes (e.g., cats vs. dogs), indicating the augmented data provided better feature representation.
