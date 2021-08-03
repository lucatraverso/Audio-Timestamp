import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
### ALL FILES IN THE SET MUST HAVE SAME SR

def explore(directorio):
    '''
    Explores a direcory printing folder and number of files
    '''
    for root, direc, files in os.walk(directorio):
        print(f'Root directory:{root} | Folders:{direc} | Files:{len(files)}')
        
def graficar(history):
    '''
    Plots Accuracy and lost of training and validation set
    Takes a the history of a model as an argument and returns None
    '''
    import matplotlib.pyplot as plt
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs = range(1, len(accuracy) + 1)
    
    plt.plot(epochs, accuracy, 'bo', label='Training acc')
    plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    return None      
#%% IMPORT FEATURES, LABELS AND SPLIT TRAIN AND TEST SET
features = np.load('features.npy')
labels = np.load('labels.npy')

features = np.expand_dims(features, -1)
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(features, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    train_size=0.8, 
                                                    shuffle=True, 
                                                    random_state=1)
batch = 20
steps = (0.8 * len(x_train)) // batch

#%% ADD LAYERS TO THE MODEL 
model = keras.models.Sequential(
    [
    Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', input_shape=(64, 101, 1), data_format='channels_last'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(128, (3, 3), activation='relu', strides=1, padding='same'),
    MaxPooling2D((2, 2), strides=2),
    Dropout(0.4),
    Flatten(),
    Dense(2048, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(3, activation='softmax')
    ]
    )
#%% COMPILE THE MODEL
model.compile(optimizer= 'adam' ,
              loss= 'categorical_crossentropy',
              metrics=['accuracy'])

#%% TRAIN THE MODEL
history = model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch,
                    epochs=10,
                    verbose=2,
                    validation_split=0.2,
                    validation_data=None,
                    shuffle=True,
                    steps_per_epoch=steps,
                    )

model.save('timestamp')
