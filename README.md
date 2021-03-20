# NextBlock - Neural Audio Distortion Repair for Realtime Network Streaming

### Live audio dropout correction using a variety of CNN-based approaches

This project aims to determine the viability of using convolutional neural networks to reduce the THD present in low-latency P2P audio communication protocols as a result of rectangular windowing artifacts. These artifacts occur when using JackTrip, when its internal ring buffer experiences packet underruns due to poor connectivity. 

The approach is to create an on-line, client-side Tensorflow prediction pipeline integrated with the [Jack Audio Connection Kit](https://github.com/jackaudio/jack2). NextBlock uses a lightweight 1D convoltional network to continuously predict the next block of incoming audio and repair the stream if packets are reported as dropped.

It also presents a solution for on-line server-side learning for realtime audio applications. Both use frequency-domain (FFT) approaches for training and inference, and require an intermediate processing step for processing the FFT before a forward pass through the network, and afterwards for the inverse, to yield the waveform prediction of the next block.

## Dataset
The [Cambridge Multitrack download library](https://www.cambridge-mt.com/ms/mtk/) provides hundreds of free multitrack recording sessions as waveform stems. I've created [this accompanying set of tools](https://github.com/carlmoore256/Cambridge-Multitrack-Dataset) to extract labels and features from the stems, and optimize the large collection of studio recorded audio tracks for machine learning purposes.

### Timbral verification of training data
Around 18 hours of audio labeled with "vocals" and/or "vox" are verified for their content via [yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet). Yamnet is trained on [AudioSet](https://research.google.com/audioset/), and provides an easy method for filtering out erroneous sounds like talking, microphone bleed and silence.

This repo is currently a work in progress!

## Model Architecture

![Model Architecture](https://raw.githubusercontent.com/carlmoore256/NextBlock/main/models/model.png)