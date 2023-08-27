# Voice Recognition System for the Mali market system

The Voice Recognition system is trained to specifically recognize the seed types used in Mali: corn, cowpea, fonio, millet, okra, peanute, rice, sesame, sorghum, and soy.

## Data Collection

The initial and crucial step in deploying a model is data collection, which involves considering two main factors: data quantity and data quality. In terms of data quantity, for fine- tuning a 10-class detector, a total of 190 samples are collected, with 10 samples per 
participant and 19 participants in total. Similarly, for testing, a total of 110 samples are collected, with 10 samples per participant and 11 participants in total. When it comes to data quality, participants were instructed to collect audio recordingsin quiet environments 
with minimal background noise, while the 10 audio recordings of the seed pronouncing the words as naturally as possible. The length of the audios varies from 1 sec to 2 sec maximum and the training participants are evenly split 10-9 male-female.

## Data Preprocessing

During the data pre-processing stage, the audio data is saved in a stan-
dardized 16 kHz 16-bit single channel PCM wave file format, ensuring consistency in the format for
subsequent analysis. Additionally, the amplitude of the audio signals is normalized to a standard
level, promoting uniformity across different recordings. As part of the data augmentation process,
five supplementary samples are generated for each audio recording. These samples introduce varia-
tions to the original audio, expanding the dataset’s diversity. One such variation is the creation of a
noise sample, achieved by adding Gaussian noise to the original recorded audio. Furthermore, lower
and higher pitch samples are derived from both the original audio and the noisy version, enabling the
model to learn representations across different pitch ranges and noise levels. Consequently, the train-
ing dataset grows to include a total of 1140 samples, while the test dataset comprises 660 samples,
allowing comprehensive evaluation of the model’s performance. Following the data augmentation
stage, an essential step in the audio processing pipeline involves computing the mel-spectrogram for each audio sample.

### Mel-Spectrogram

The mel-spectrogram provides a visual representation of the audio signal’s fre-
quency content over time. This is achieved through a series of steps, beginning with the application of
a Short-Time Fourier Transform (STFT) to the audio signal. The STFT transforms the audio from
the time domain to the frequency domain, revealing the frequency components present at different
time frames. From the resulting STFT output, the magnitude spectrum is computed, representing
the intensity of each frequency component. Then, the mel filterbank is applied to the magnitude
spectrum. The mel filterbank consists of a set of triangular filters that are evenly spaced on the mel
scale, which approximates the perceptual scale of human hearing. This filtering process emphasizes
the energy in different frequency regions based on the filters’ shape. Following the filtering stage,
the logarithmic compression is typically applied to the resulting filterbank energies. This logarithmic
scaling reduces the dynamic range of the values and aligns them with human perception. Finally,
for consistency and further analysis, the mel-spectrogram undergoes additional normalization tech-
niques, such as mean subtraction or standardization. In summary, the computed mel-spectrogram
provides a two-dimensional representation of the audio signal, with time on the x-axis, frequency on
the y-axis, and the intensity or magnitude of the energy represented through color or shading. The
mel-spectrogram representations are then saved as images ready to be fed to the image classification
model.


<figure>
  <p align="center">
  <img src="https://github.com/simoneVU/Voice-Recognition-Mali-Seeds/blob/main/images/mel_spectrograms.png" width="400" title="1D audio signal vs 2D melspectrogram">
    </p>
</figure>

On the left side of the image above there are the 1D audio signals while on the right side their respective 2D melspectrogram.

## Model Implementation

The model backbone is implemented using the pre-trained model ConvNeXt small from the timm library 1. The pre-trained model head is switched for a 10-class one,
and then fine tuned for the 10 seed classes. More specifically, the 10-class head consists of a batch
normalization layer, a linear layer followed by ReLU, another batch normalization layer followed
by dropout with probability 0.5 and the output layer (see figure 5). The model is fine tuned and
validated over 300 epochs on 1000 and 100 voice recordings respectively. Hence, as it is possible to
see from figure 5, the backbone weights are frozen, while the weights of the model head are to be
optimized. After fine tuning the model head, the whole model is tested on 660 voice recordings. The model architecture can be seen in the following image:


<figure>
  <p align="center">
  <img src="https://github.com/simoneVU/Voice-Recognition-Mali-Seeds/blob/main/images/ICT4D_model_head.png" width="700" title="model architecture">
    </p>
</figure>

While the final result seen as a confusion matrix are the following:

<figure>
  <p align="center">
  <img src="https://github.com/simoneVU/Voice-Recognition-Mali-Seeds/blob/main/images/confusion_matrix_1_f17d6e42908696b83fe0.png" width="700" title="final CF">
    </p>
</figure>

# Reproducibility

In order to reproduce the results above run the code in collab notebook https://github.com/simoneVU/Voice-Recognition-Mali-Seeds/blob/main/Voice%20Recognition%20model.ipynb. Email me at s2.colombo@student.vu.nl for the data collected.
