import librosa # audio analysis package : https://librosa.org/doc/latest/index.html 
import pandas # to extract any data feautures : https://pandas.pydata.org/ 

import matplotlib.pyplot as plt # graph plotting (like the frequency, in this case)
import glob
import numpy

dataset = pandas.DataFrame() # To 

from sklearn.model_selection import train_test_split

#Form some variable

genres = ["blues", "classical", "country","disco", "hiphop", "jazz", "metal","pop","reggae", "rock"]

for genre in genres:
	print(genres)
	for filename in glob.glob() #GLOB IS TO EXTRACT 
	# * IS EVERYTHING INSIDE THE FOLder, wav is the extension
	# print to check if it is working correctly
	# doing the same thing 
	# GOOGLE : MANY FEAUTURES THAT CAN 
	x, sr = librosa.load(filename, sr = 44100)
	spectral_centroids = librosa.feature.spectral_centroids(x, sr=sr)
	# [0] - multiple features in the array and we need JUST THE FIRST LINE
	zero_crossing = librosa.feature.zero_crossing_rate(x)
	# zero is how many times it crosses zero. 
	#Syntax is very similar
	chromagram = librosa.feature.chroma_stft(x, sr=sr)
	mfcc = librosa.feature.mfcc(x, sr=sr) # 20 features ! HUMAN VOICE DETECTION
	rms = librosa.feature.rsm(x) #rms (root mean square - quadratic average)

	dataset = dataset.append({
		"filename": filename,
		"spectral_centroids":numpy.mean(spectral_centroids)
		"spectral_rolloff":numpy.mean(spectral_rolloff)
		"bandwith":numpy.mean(spectral_centroids)
		"zero_crossing":numpy.mean(zero_crossing)
		"chronogram": numpy.mean(chronogram)
		"rms":numpy.mean(rms)
		""
		'genre':genre

		# WHY ONLY ONE '' INSTEAD OF "" ?

		# ctrl d to SELECT SAME THING

		}, ignore_index = True)


["filename","spectral_centroids","bandwith","zero_crossing","chronogram","rms"]

# QUESTIONS: HOW TO COPY THE NAME OF THE VARIABLES?s 

dataset = dataset[["      "]] # will SORT according to this ORDER

dataset.to_csv("dataset.csv")

# crtl + d : select multiple 
# multiple cursors 