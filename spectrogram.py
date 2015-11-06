import convertToWav
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal

import matplotlib.pyplot as plt
import pylab


class SubjectWav:
	def __init__(self, wav_file):
		self.wav_file = wav_file
		self.sample_rate = None
		self.freqency_resolution = 1300 #uknown measure of frequency resolution, higher number, larger resolution
	def __load_data__(self):
		if self.sample_rate is None:
			fs, frames = wavfile.read(self.wav_file)
			self.sample_rate = fs
			self.audio_data = frames
	def get_spectrogram(self, at_second, for_seconds=4,channel=0):
		self.__load_data__()
		window_start = at_second * self.sample_rate	
		window_size = for_seconds * self.sample_rate
		f, t, Sxx = signal.spectrogram(self.audio_data[window_start:window_start + window_size,0],
					  self.sample_rate,
					  nperseg = self.freqency_resolution,
					  ) 
				#to decibel
		return f, t, 20.*np.log10(np.abs(Sxx)/10e-6)
	


def plotSubjectWav(sw, at_second,for_seconds=4):
	f, t, Sxx = sw.get_spectrogram(at_second,for_seconds)
	plt.pcolormesh(t, f[0:300], Sxx[0:300])
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

"""
	itearates through all wav files, executing func and passing it the path to wav file
"""
def iterateThroughWav():
	for wav in os.listdir(convertToWav.VIDEO_ROOT):
		if wav.endswith(".wav"):
			full_path = convertToWav.VIDEO_ROOT + "\\" + wav
			yield full_path

if __name__ == "__main__":
	sw = None
	for p in iterateThroughWav():
		sw = SubjectWav(p)
	plotSubjectWav(sw, 31*60+17)


