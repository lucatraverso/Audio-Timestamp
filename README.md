# Timestamp audio file with 4 labels

## Instructions

Execute *data_preparation.py* in the same directory as the MUSAN dataset or any other dataset that has separated music, noise and speech. 
This script will return two *.npy* files corresponding to the features and labels.
Then run *model.py*. This will configure, train and save the model.
*predicting.py* processes the example file and plots the first 40 seconds of the result. It saves the result as an image.

## Code and Model

I developed my model using an article as a reference[1]. 
The algorithm uses a Convolutional Neural Network that takes a Mel spectrogram as input and recognizes music, noise and speech. The silence detection was carried out using a threshold.

### *data_preparation.py*

Explores a directory and processes audio files in it. 
Split each file into N fragments of 1 second, and then computes a Mel spectrogram for each fragment. This results in N spectrograms with 64 frequency bins and 101 time frames. 
Organizes all spectrograms in numpy arrays. 
The dataset must be in the working directory as well and divided into music, noise and speech. Saves features in *features.npy* and labels in *labels.npy*.

### *model.py*

Configure, compile, train and save the model.

The convnet architecture is the following: 3 convolutional layers with 3x3 filters, stride 1 and pooling to maintain the input size. Each layer had 32, 64 and 128 filters respectively and was followed by a 2x2 Max Pooling layer with stride 2.
The convolutional layers were followed by 2 fully connected layers of 2048 and 1024 units each. All layers had a ReLU as activation function.
The final layer was a softmax with 3 outputs.

## Model Summary:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d_6 (Conv2D)            (None, 64, 101, 32)       320       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 32, 50, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 32, 50, 64)        18496     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 16, 25, 64)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 16, 25, 128)       73856     
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 8, 12, 128)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 12, 128)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 12288)             0         
_________________________________________________________________
dense_6 (Dense)              (None, 2048)              25167872  
_________________________________________________________________
dense_7 (Dense)              (None, 1024)              2098176   
_________________________________________________________________
dense_8 (Dense)              (None, 3)                 3075      
=================================================================
Total params: 27,361,795
Trainable params: 27,361,795
Non-trainable params: 0
_________________________________________________________________
```

The algorithm was trained in 10 epochs with a fragment of the MUSAN dataset (550 MB  out of an 11GB dataset). The entire dataset was not used due to time and hardware limitations.

![acc](https://user-images.githubusercontent.com/26192412/116941844-dd92b680-ac46-11eb-9434-a2871499fa05.png)
![loss](https://user-images.githubusercontent.com/26192412/116941846-dec3e380-ac46-11eb-883c-ba6af9610bdb.png)

```
Testing Loss:  0.0886
Testing Accuracy: 0.9738
```

The training and validation accuracy does not show signs of overfitting. 

### *predicting.py*

To stamp an external file, the sample rate is reduced to 16000 Hz so the spectrogram fits in the model input. 
The model returns an array of 3 values, each corresponding to the probability of the audio frame to belong to each class (music, noise and speech).  
This results in an array of N rows (if the audio is N seconds long) and 3 columns.

The values are rounded to 1 if are higher than 0.7 or 0 if not. 
At this stage a column corresponding to silences calculated (using threshold) and added to the prediction matrix.

[Audio](images/40sec.png)

![40sec](https://user-images.githubusercontent.com/26192412/116942312-bbe5ff00-ac47-11eb-89b5-0bdd6960a6cf.png)

As it is observable, the tag does not match with the correct values in every frame but it is improvable using more training data. Another aspect to improve is precision. The model only process 1 second frames. This may be perfectible using an envelope detector that indicates when was an abrupt change in the sound level. 

## Aspects to improve
- Train the algorith with more data.
- Explore other ways to store data rather than numpy arrays that may be more lightweight.
- Increase the time precision using a level detector in the border frames (frames that correspond to one class and are next a different class).
- Results presentation

### References
[1] Jang, Byeong-Yong, et al. "Music detection from broadcast contents using convolutional neural networks with a Mel-scale kernel." EURASIP Journal on Audio, Speech, and Music Processing 2019.1 (2019): 1-12.
