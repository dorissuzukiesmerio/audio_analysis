import librosa
# python -m pip install librosa before ! On cmd
import numpy 

import 
import sklear

# audio_file = ".genres/classical/classical.00000.wav"  #to load the file. Notice that every file is in the same format
# # Be attentive to the version that you are using on librosa !
# # sr = sample rate - in one second, how many 
# x, sr = librosa.load(audio_file, sr=44100)


# FFT - Time-Frequency-View
# https://commons.wikimedia.org/wiki/File:FFT-Time-Frequency-View.png
# frequency - distance between two peaks
# magnitude - "height"
# 
# sin curve

# composition vs. decomposition 
# like multiplying two prime numbers vs. getting a number and then trying to figure our 
# encryption - rely on that fact. Bitcoins

# frequency scale is easier for computer to read

# wave_plot = plt.figure(figsize=(13,5))
# librosa.display.waveplot(x, sr=sr) # THis is the usual graph we see on our recordings: time and frequency 
# wave_plot.savefig("waveplot_classical0000.png")
# plt.close()

# To do a loop:

genre = "classical"
num = "00000"

audio_file = ".genres/" + genres + "." + num + ".wav"
x, sr = librosa.load(audio_file, sr=44100)


wave_plot = plt.figure(figsize=(13,5))
librosa.display.waveplot(x, sr=sr) # THis is the usual graph we see on our recordings: time and frequency 
wave_plot.savefig("waveplot_" + genre + num + .png")
plt.close()

stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))
spectrogram = plt.figure(figsize=(13,5)) # This is the graph like looking that picture from above
librosa.display.specshow(stft_data_db, sr=sr, x_axis="time", y=axis='hz')
spectogram.savefig(wave_plot.savefig("spectogram_" + genre + num+ .png")


# Some people transform audio into picture and then analyse the picture - if you google

# to average the picture
# Intensity - like color in the spectogram
# frequency
# weighted average of the frequency, weighted by the intensity (color)

spectral_centroids = librosa.feature.spectral_centroids(x, sr=sr)[0]
spectral_centroids_plot = plt.figure(figsize = (13,5))
frames = range(len(spectral_centroid))
t = librosa.frames_to_time(frames)
librosa.display.waveplot(x, sr=sr, alpha=0.3) # THis is the usual graph we see on our recordings: time and frequency 

# sklearn.preprocessing.minmax_scale(spectral_centroids, axis=0)

# plt.plot(t,								, color='r')


plt.plot(t, sklearn.preprocessing.minmax_scale(spectral_centroids, axis=0), color='r')


# THIS CURVE IS A SUMMARY STATISTICS OF THE AUDIO FILE

#QUESTIONS:
# choice of number
# alpha

# ROLLOFF:
# Level of frequency. 
# 85% of intensity will be bellow a certain frequency

spectral_rolloff = librosa.feature.spectral_centroids(x, sr=sr)[0]
spectral_rolloff_plot = plt.figure(figsize = (13,5))
spectral_rolloff_plot.savefig("spectral_rolloff" + genre + num + ".png")
plt.close()

# New class: 


# upload 

spectral_rolloff = librosa.feature.pectral_rolloff(x+0.01, sr=sr)[0] # 0.01 is to fix a problem. rolloff has problem in x=0, so we shift everything by a little bit



