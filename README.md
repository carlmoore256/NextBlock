# NextBlock
Live audio dropout correction using a variety of CNN-based approaches

This repo is currently a work in progress!

This project aims to determine the viability of using convolutional neural networks to reduce the THD present in low-latency audio communication protocols, specifically JackTrip. Our approach is to create an on-line, client-side Tensorflow prediction pipeline & Jack (audio backend) interface, using a lightweight 1D convoltional network to fill gaps in the stream as they occur for the client. It also presents a solution for on-line server-side learning for realtime audio applications. Both use frequency-domain (FFT) approaches for training and inference, and require an intermediate processing step for processing the FFT before a forward pass through the network, and afterwards for the inverse, to yield the waveform prediction of the next block.
