# Hands-On Implementation on Convolutional Neural Network (CNN)
    # Implementation of basic CNN to classify images based on their respective classes

### About the Dataset
'''
- Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of **60000** examples and a test set of **10,000** examples. 
- Each example is a **28x28** grayscale image, associated with a label from **10 classes**. 
- Ten classes associated with the labels are:
    - Label 1: T-shirt/top
    - Label 2: Trouser/pants
    - Label 3: Pullover shirt
    - Label 4: Dress
    - Label 5: Coat
    - Label 6: Sandal
    - Label 7: Shirt
    - Label 8: Sneaker
    - Label 9: Bag
    - Label 10: Ankle boot
- Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. 
- It shares the same image size and structure of training and testing splits.

### Dataset Information

- Each image is **28 pixels** in height and 28 pixels in width, for a total of **784 pixels** in total.
- Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between **0 and 255**.
- The training and test data sets have **785 columns**.
- The **first** column consists of the **class labels** and represents the **article of clothing**.
- The **rest of 784 columns** (785-1) contain the **pixel-values** of the associated image.
'''

# Import the necessary libraries
import numpy as np                                                      # Basic Python library
import pandas as pd
import matplotlib.pyplot as plt                                         # For visualization
from sklearn.model_selection import train_test_split                    # For splitting the dataset
import tensorflow as tf                                                 # Tensorflow framework for building Neural Network
from tensorflow import keras                                            # Importing keras library from tensorflow
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D  # Importing convolution, Flatten and Dense Layers
from keras.models import Sequential                                     # Importing Sequential layer from keras
from keras.models import load_model                                     # Loads the saved model
from sklearn.metrics import accuracy_score

###-----------------------------------------------------------------------------------------------------------------------------------------

# Dataset Loading
(X_train, y_train),(X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()   # Loading the fashion_mnist dataset
print('Training Data Shape:', X_train.shape, y_train.shape)                         # Printing shapes of train dataset
print('Test Data Shape:', X_test.shape, y_test.shape)                               # Printing shapes of test dataset
print('-'*100)

###-----------------------------------------------------------------------------------------------------------------------------------------

# Data Analysis

## Find the unique numbers from the train labels
classes = np.unique(y_train)                                   # Finding the unique values in y_train
n_class = len(classes)                                         # Identifying the total number of classes
print('Total number of outputs : ', n_class)                   # Displays total number of classes
print('Output classes : ', classes)                            # The number of classess are mentioned at the top for reference
print('-'*100)

## Displaying sample train images
img_num = list(np.random.randint(1, X_train.shape[0], 5))      # Getting 5 random numbers from total train images
plt.figure(figsize=(10,10))                                    # Setting the figure size

for i,j in enumerate(img_num):                                 # Iterating over the selected 5 random images
    plt.subplot(1, len(img_num), i+1)                          # Setting the subplot for each image
    plt.imshow(X_train[j], cmap='gray')                        # Displaying the plot, cmap can be 'greens', 'reds', 'blues', 'rgb' (try changing the 'gray' with the ones mentioned here)
    plt.title("Label: {}".format(y_train[j]))                  # Setting the title of the plot with the labels
    plt.axis('off')                                            # Hiding the axis
    
plt.show()

###-----------------------------------------------------------------------------------------------------------------------------------------

# Data Preprocessing

# Reshaping the images to get the number of channels (both train & test)
X_train = X_train.reshape(-1, 28,28, 1)
X_test = X_test.reshape(-1,28,28, 1)
print('Reshaped Train & Test dataset: ', X_train.shape, X_test.shape)
print('-'*100)

# Normalizing the dataset
X_train = (X_train/255).astype('float32')                       # Converting the values from integer to float and rescaling 
X_test = (X_test/255).astype('float32')                         # it between (0-1) to avoid outlier problems

# Checking the max and min values after scaling
print("Max and Min value in X_train:", X_train.max(), X_train.min())
print("Max and Min value in X_test:", X_test.max(), X_test.min())
print('-'*100)

# Convert the target feature to one-hot vectors
y_train_onehot = pd.get_dummies(y_train)                        # This will convert the lables column from 1 to total number of labels 
y_test_onehot = pd.get_dummies(y_test)                          # (10 for this dataset)

print("Shape of y_train:", y_train_onehot.shape)
print("One value of y_train:", y_train_onehot)
print('-'*100)

###-----------------------------------------------------------------------------------------------------------------------------------------

# Data Split
# Let us try to avoid test data from getting exposed during training
# Doing Train-Validation split 
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)
print('Train Dataset Shape: ', X_train_.shape, y_train_.shape)
print('Validation Dataset Shape: ', X_val.shape, y_val.shape)
print('-'*100)

###-----------------------------------------------------------------------------------------------------------------------------------------

# Model Building

#####---------------------------------
# Initializing basic CNN model

basic_model = Sequential()                                                                          # Setting the sequential model for adding further layers
basic_model.add(Conv2D(filters=28, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))      # First convolution layer
basic_model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))                               # Adding all the other convolution layers
basic_model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))                               # with 'relu' activation which is best for all
basic_model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))                               # the middle CNN layers
basic_model.add(Flatten())                                                                          # Flattening the CNN layers
basic_model.add(Dense(64, activation="relu"))                                                       # Adding dense layers for further training
basic_model.add(Dense(128, activation="relu"))
basic_model.add(Dense(10, activation="softmax"))                                                    # Final layer with '10' classes as output
                                                                                                    # Uses 'softmax' activation for better classification

# Model Compilation
basic_model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")        
# Model Summary
print(basic_model.summary())        

# Model_Fitting
# Batch size and epochs can be varied based on the need/performance
tr_history = basic_model.fit(x=X_train_, y=y_train_, batch_size=64, epochs=3, validation_data=(X_val, y_val))  

# Checking final validation loss and accuracy
base_val_acc = basic_model.evaluate(X_val, y_val)
print(basic_model.evaluate(X_val, y_val))

## Accuracy and Loss plots
accuracy      = tr_history.history['accuracy']                      # Extracting metrics like accuracy & loss from training history
val_accuracy  = tr_history.history['val_accuracy']
loss          = tr_history.history['loss']
val_loss      = tr_history.history['val_loss']
epochs        = range(len(accuracy))                                # Get number of epochs
# Train Plot
plt.plot  (epochs, accuracy, label = 'training accuracy')
plt.plot  (epochs, val_accuracy, label = 'validation accuracy')
plt.title ('Training and validation accuracy')
plt.legend(loc = 'lower right')
plt.savefig('plots/Train-Val_ACC_Curve.png', dpi=120)
plt.figure()
# Validation Plot
plt.plot  (epochs, loss, label = 'training loss')
plt.plot  (epochs, val_loss, label = 'validation loss')
plt.legend(loc = 'upper right')
plt.title ('Training and validation loss')
plt.savefig('plots/Train-Val_Loss_Curve.png', dpi=120)
plt.figure()

# Saving the model
basic_model.save('model/basic_cnn.h5')
basic_model.save_weights('model/basic_cnn_weights.h5')



###-----------------------------------------------------------------------------------------------------------------------------------------

# Loading the saved model
# Basic model
basic_cnn = load_model('model/basic_cnn.h5')
basic_cnn.load_weights('model/basic_cnn_weights.h5')


###-----------------------------------------------------------------------------------------------------------------------------------------

# Predicting and vizualizing the test image with basic model
n = np.random.randint(1, X_test.shape[0])                       # Getting the random value from the entire test imges

plt.title(y_test[n])
plt.imshow(X_test[n])
y_pred = basic_cnn.predict(X_test[n].reshape(1,28,28,1))
print("Softmax Outputs:", y_pred)
print(y_pred.sum())

bm_actual_label = y_test[n]
bm_pred_label = np.argmax(y_pred)

# Convert the predicted probabilities to labels
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   ## This is the order in which the dataset is read
for i in y_pred:
    for j, k in enumerate(i):
        if k == y_pred.max():
            print('Predicted_Label:', labels[j])



# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\n BM_Validation_Loss_Accuracy = {base_val_acc}, BM_Actual_Label = {bm_actual_label}, BM_Predicted_Label = {bm_pred_label}.')
