# Introduction
In this project, we explore audio classification techniques with a vehicular noise dataset to identify time-critical emergency vehicles approaching potential areas 
of congestion, like junctions, to ease the congestion.
The objective of this project is to build a model that will be able to distinguish between the sound of the sirens, the emergency vehicles, and the sound generated 
by the other vehicles. This project follows two main techniques, them being time domain analysis and frequency domain analysis. The classification is done
using convolutional neural networks (CNN) and long-short-term memory (LSTM).

# Methodology

**Initial Data Analysis:** We will be utilizing a dataset which is a
collection of sound clips of emergency vehicle siren and
non-emergency vehicle sound. We will first read the audio data, then
we will explore and preprocess the raw audio data. We have used a
python library called Librosa for raw audio manipulation. Librosa is
an open-source library in python that is used for audio processing and
analysis.

**Dataset:** The dataset used for this model training contains two files,
one the emergency vehicle sound, and the other non-emergency
vehicle sound. These two files contain multiple audio clips, recorded
at different places. 

Data Source : [Link](https://drive.google.com/file/d/14ZCIdME9M1CN7Bu7MAbZ5qY9X0HWr0nC/view)

**Data Preparation:** The data preparation for this project revolves
around the technique of breaking raw audio into chunks. Short two-second chunks are obtained by a sliding window over the input
creating chunks which will give us the samples for training. We have
split the audio clips into audio chunks of two seconds or 32000
samples. The audio data which is an array is the number of samples
which is set to 32000 and the sampling rate which is set to 16000.

**Implementation of CNN:** We are using one-dimensional
convolutional layers or Conv1D because we want the filters to move
across one dimension only. The number of filters in the first
Conv1D layer is 8, and the height of the filters is 13. The number of
filters in the second Conv1D layer is 16 and the height of the filter is
11. then comes down the GLobal max-pool layer, which will convert
the two-dimensional array output of the previous layer. The
one-dimensional output will be fed to one dense layer.
Finally, the final layer makes the prediction.

**Spectrogram and Frequency domain analysis:** Spectrogram accepts
the raw audio wave and then breaks it into chunks or windows and
then applies fast Fourier transformation(FFT) on each window to
compute the frequencies. Using the scipy.signal.spectrogram, we
obtain an array ‘spec’ which contains the required spectrogram. This
is done on the same chunks used in the time domain analysis. The
spectral analysis is expected to produce better results as frequencies of
sirens are expected to stand out prominently in the sample and easier
to identify than sirens in time domain samples where we can see only
repeating peaks in amplitude which can easily be disturbed with
generic noise in traffic


![image](https://github.com/user-attachments/assets/ea98a7c9-cff4-4def-b6f3-627c251a033b)
