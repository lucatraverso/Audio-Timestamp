import os
import numpy as np
import librosa, librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
def create_spectrogram(audio, n_fft, hop, win, normalize=False):
    '''
    Creates an spectrogram.
    Input: 
          audio: array.
          n_fft: int (fft length).
          hop: int (hop size).
          win: int (window size)
          normaliza: bool (optional)
    Out: Matriz, un espectrograma logaritmico (con valores pasados a dB)
    '''
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=win)
    logspec = librosa.amplitude_to_db(np.abs(stft))
    if normalize == True:
        return logspec / logspec.max()
    else: 
        return logspec

def mel_spectrogram(audio, sr, nmels, nfft, hop, win):
    '''
    Creates a mel spectrogram.
    Input: 
          audio: array.
          sr: int (sample rate)
          n_fft: int (fft length).
          hop: int (hop size).
          win: int (window size)
    Out: Matriz, un espectrograma logaritmico (con valores pasados a dB)
    '''
    melspec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=nmels, 
                                             n_fft=nfft, hop_length=hop, 
                                             win_length=win)
    logmelspec = librosa.amplitude_to_db(melspec)
    return logmelspec

def list_spectrograms(directory, nfft, hop, win, normalize=False, mel=True):
    '''
    For all files in a path and subdirectories:
        -Imports the audiofile
        -Divide it into fragments of 1 sec
        -Creates spectrograms from those fragments
        -Add them to a list
    In: Path: string, directory.
    Out: Numpy array containing spectrograms
    nfft, hop, win Spectrogram parameters
    normalize only works when using regular spectrograms (mel=False)
    
    '''
    spectrograms = []
    print('Starting listing...')
    for root, direc, files in os.walk(directory):
        print(f'Directorio raiz:{root} | Carpetas:{direc} | Archivos:{len(files)}')
        for name in files:
            filepath = os.path.join(root, name)
            if filepath[-3:] == 'wav':
                audio, sr = librosa.load(filepath, sr=None)
                audio_splitted = split_audio(audio, sr)
                for arr in audio_splitted:
                    if mel == True:
                        spec = mel_spectrogram(arr, sr, 64, nfft, hop, win)
                    else:
                        spec = create_spectrogram(arr, nfft, hop, win, normalize)
                    spectrograms.append(spec)
    return np.array(spectrograms)

def split_audio(audio, fs):
    '''
    Divides an audiofile into 1s fragments.
    In: 
        audio: array
        fs: int, sample rate
    Out: Arrays of 1 sec audios
    '''
    n = len(audio)
    splitted_audios = []
    audio = pad(audio, fs)
    cant = int(n/fs)
    for i in range(1, 1 + cant):
        start = fs*(i-1)
        end = fs*i
        one_sec_audio = audio[start:end]
        splitted_audios.append(one_sec_audio)
    return np.array(splitted_audios)

def pad(audio, fs):
    '''
    Zeropad an audiofile to have an integer length
    '''
    N = len(audio)
    zeros_to_add = fs - (N % fs)
    zeros = np.zeros(zeros_to_add)
    audio = np.append(audio, zeros)
    return audio

def plotSpecs(specs_arr):
    '''
    Plotea 5 espectrogramas.
    In: Array [5, Height, Widht]
    '''
    fix, axes = plt.subplots(1, 5)
    axes = axes.flatten()
    for spec, ax in zip(specs_arr, axes):
        ax.pcolormesh(spec)
    #plt.colorbar()
    plt.show()

#%%
def main():
    directory = os.path.join('.', 'short_ds')

    music_dir = os.path.join(directory, 'music')
    noise_dir = os.path.join(directory, 'noise')
    speech_dir = os.path.join(directory, 'speech')
    
    music = list_spectrograms(music_dir, 512, 160, 400, normalize=False, mel=True)
    noise = list_spectrograms(noise_dir, 512, 160, 400, normalize=False, mel=True)
    speech = list_spectrograms(speech_dir, 512, 160, 400, normalize=False, mel=True)
    features = np.concatenate((music, noise, speech))
    
    l = [[0], [1], [2]]
    labels = np.array(music.shape[0] * l[0] + noise.shape[0] * l[1] + speech.shape[0] * l[2])
    
    return features, labels
    
def export_data(features, labels):
    np.save('features', features, fix_imports=False)
    np.save('labels', labels, fix_imports=False)
    
if __name__ == '__main__':
    features, labels = main()
    export_data(features, labels)