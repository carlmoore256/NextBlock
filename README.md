# NextBlock - Low Latency Audio Distortion Reduction
## Live audio dropout correction using a variety of CNN-based approaches

This project aims to determine the viability of using convolutional neural networks to reduce the THD present in low-latency audio communication protocols as a result of rectangular windowing artifacts. These artifacts occur when using JackTrip, when its internal ring buffer experiences packet underruns from poor connectivity. 

The approach is to create an on-line, client-side Tensorflow prediction pipeline integrated with the [Jack Audio Connection Kit](https://github.com/jackaudio/jack2). NextBlock uses a lightweight 1D convoltional network to continuously predict the next block of incoming audio and repair the stream if packets are reported as dropped.

It also presents a solution for on-line server-side learning for realtime audio applications. Both use frequency-domain (FFT) approaches for training and inference, and require an intermediate processing step for processing the FFT before a forward pass through the network, and afterwards for the inverse, to yield the waveform prediction of the next block.

This repo is currently a work in progress!


![Model Architecture](https://raw.githubusercontent.com/carlmoore256/NextBlock/main/models/model.png)