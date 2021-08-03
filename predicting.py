from tensorflow import keras
from data_preparation import split_audio, mel_spectrogram
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import librosa
from librosa import display
import matplotlib.pyplot as plt

def prepare_audio(path):
    '''
    Prepares an audiofile for prediction.
    Splits in one second fragments and compute mel spectrograms.
    
    In: str - filepath
    Out: numpy array of shape (N, 64, 101)
        (N: length of audiofile in seconds)
    '''
    spectrograms = []
    audio, sr = librosa.load(path, sr=16000)
    
    audio_splitted = split_audio(audio, sr)
    for arr in audio_splitted:
        spec = mel_spectrogram(arr, sr, 64, 512, 160, 400)
        spectrograms.append(spec)
    return np.array(spectrograms)

def evaluate(model):
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

    return model.evaluate(x_test, y_test, verbose=2)

def binary(lista, ts):
    '''
    Takes a list and a threshold and rounds all numbers to 0 or 1
    '''
    for i, j in enumerate(lista):
        if j >= ts:
            lista[i] = 1
        else:
            lista[i] = 0
    return lista

def plot(audio, sr, music, noise, speech, silence, save=False, filename='image.png'):
    N = range(len(music))
    plt.figure(figsize=(10, 2))
    display.waveplot(audio, sr, linewidth=0.5)
    plt.step(N, music, label='Music', linewidth=1)
    plt.step(N, noise, label='Noise', linewidth=1)
    plt.step(N, speech, label='Speech', linewidth=1)
    plt.step(N, silence, label='Silence', linewidth=1)
    plt.legend()
    if save == True:
        plt.savefig(filename, dpi=300)

def rootmeansquare(lst):
    '''
    Computes the root mean square value of an array
    '''
    return np.sum((lst**2))/len(lst)

def detect_silence(audio, sr, ts):
    '''
    Detects fragments of silence in the signal via threshold
    In: audiofile, samplerate and threshold
    Out: binary array indicating silence fragments
    '''
    silences = []
    audio_splitted = split_audio(audio, sr)
    for arr in audio_splitted:
        #rms = librosa.feature.rms(arr, frame_length=sr, hop_length=sr)
        rms = rootmeansquare(arr)
        if rms > ts:
            silences.append(0)
        else:
            silences.append(1)
    return np.array(silences)

if __name__ == '__main__':
    model = keras.models.load_model('timestamp') #LOADING MODEL
    file = 'dsp-ml-challenge.wav' #FILEPATH
    
    audio, sr = librosa.load(file, sr=None) #LOAD FILE
    #PREPARE FILE FOR PREDICTION - DIVIDES INTO 1sec FRAGMENTS
    #COMPUTES MEL SPECTROGRAMS
    test_audio = prepare_audio(file) 
    test_audio = np.expand_dims(test_audio, -1)
    
    prediction = model.predict(test_audio, verbose=1)
    
    labels = ['Music', 'Noise', 'Speech', 'silence']
    
    ### DETECTS SILENCE AND APPENDS IT TO PREDICTION
    silence = detect_silence(audio, sr, 0.000001)
    silence = np.expand_dims(silence, -1)
    prediction = np.append(prediction, silence, axis=1)
    prediction = prediction.round(0)
    
    #CREATES 3 BINARY SIGNALS INDICATING WHAT LABEL CORRESPOND TO EACH SECOND
    music = binary(prediction[:,0], 0.7)
    noise = binary(prediction[:,1], 0.7)
    speech = binary(prediction[:,2], 0.7)
    
    ### STARTING AND ENDING POINT TO PLOT
    
    start = 0
    end = 40
    plot(audio[start:sr*end], 
         sr, 
         music[start:end], 
         noise[start:end], 
         speech[start:end], 
         silence[start:end], 
         True, 
         filename='40sec.png')
    
