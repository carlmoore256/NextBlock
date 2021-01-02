# ----- NEXTBLOCK DATA GENERATOR -----

# ============================================================================
# Example generator designed for the chunk-based cambridge dataset 
# see dataset.py for more information on the dataset designed for this model
# 
# DATA PRE-PROCESSING STEPS:
# 
# Fast Fourier Transform (FFT) is computed for a pair of waveform examples: 
# x = audio clip[index : index + block_size]
# y = audio clip[index + offset : index + + offset + block_size]
# 
# where offset = (block_size/hop_ratio) * y_offset
# and, index = random index within an audio clip
# 
# FFT example/label pairs are NOT converted to power spectrum, as they
# typically are in feature extraction applications. Instead, the complex
# form is preserved to properly represent phase information, which is critical
# in this case.
# 
# In order to prevent convolution edge artifacts from interfering with 
# training, the FFT examples are offset so that the lowest bins are interlaced
# and mirrored about the center. The process is reversed when converting back 
# into waveform audio after inference. This may possibly also improve the 
# gaussian distribution of salient information within the network,
# since typically, most of the useful data in an FFT is in the lower bins.
# This needs to be experimentally verified, however.
# 
# Typically log-transforms like the Mel and Bark scales (and MFCCs) are used
# to reduce dimensionality and amplify useful frequency information.
# However, in this case, we are most concerned about re-constructing proper 
# phase and magnitude information, which would be lost using a log-transform.
# 
# See de_center_data() and center_data() for details about this
# pre-processing step.
# 
# The rectangular [x + y(i)] representation of the computed FFT tensors is
# converted into a float32 tensor, containing the real and imaginary
# components along the last axis represented to the network as two 
# separate channels.
# 
# X and y pairs are fed to the network, which minimizes an MSE regression
# loss, or the re-construction of an unseen block of audio which follows x.
# ============================================================================

import tensorflow as tf
import numpy as np

class DataGenerator():
    def __init__(self, batch_size, block_size, channels, data, hop_ratio=2, y_offset=1, normalize=True):
      self.batch_size = batch_size
      self.block_size = block_size
      self.channels = channels
      self.hop = block_size//hop_ratio
      self.hop_ratio = hop_ratio
      self.data = data
      self.win = tf.signal.hann_window(self.block_size)
      self.y_offset = y_offset
      self.dataset_index = 0
      self.num_examples = self.data.shape[0]
      self.normalize=normalize

    def load_audio(self, dir):
      # tensorflow read file (can read any file)  
      raw_audio = tf.io.read_file(dir)
      # decodes raw_audio into -1f to 1f float32 tensor
      waveform = tf.audio.decode_wav(raw_audio, desired_channels=1)
      # waveform[0]: float32 tensor of waveform data
      # waveform[1]: samplerate (ignoring sr for now)
      return waveform[0]

    # waveform audio -> FFT (tf.complex64 dtype)
    def fft(self, audio):
      fft = tf.signal.fft(audio)
      if self.normalize:
        fft = self.normalize_fft(fft)
      return fft

    def normalize_fft(self, fft):
      scalar = 1.0/self.block_size
      normalized_fft = tf.math.multiply(fft, scalar)
      return normalized_fft

    def reverse_normalize_fft(self, normalized_fft):
      return normalized_fft * self.block_size

    # x + y(i) -> magnitude, angle
    def rectangular_to_polar(self, rectangular):
      magnitude = tf.abs(rectangular)
      angle = tf.math.angle(rectangular)
      polar = tf.concat([magnitude, angle], axis=2)
      return polar

    # loads a single file as an FFT tensor to be fed into the net
    def load_single(self, file, window=True):
      audio = self.load_audio(file)
      print(f'audio shape {audio.shape}')
      frames = tf.squeeze(tf.signal.frame(audio, self.block_size, self.hop, axis=0))
      if window:
        frames *= self.win
      tensor = self.frames_to_fft_tensor(frames, window=window)
      return audio, frames, tensor

    # float32 tensor to rectangular notation:
    # [real, imaginary] -> [complex,]
    def complex_to_ri(self, tensor):      
      real = tf.math.real(tensor)
      imag = tf.math.imag(tensor)
      ri_t = tf.concat([real, imag], axis=2)
      return ri_t

    # rectangular notation to float32 tensor
    # [complex,] -> [real, imaginary]
    def ri_to_complex(self, tensor):      
      real = tensor[:,:,0]
      imag = tensor[:,:,1]
      # account for FFT mirror cutoff at N/2+1
      mirror = tf.zeros_like(real)
      real = tf.concat([real,mirror], axis=1)
      imag = tf.concat([imag,mirror], axis=1)
      complex_t = tf.dtypes.complex(real, imag)
      return complex_t

    def overlap_add(self, frames):
      audio = tf.signal.overlap_and_add(frames, self.hop)
      return audio

    # prediction in complex notation -> audio tensor
    def ifft_prediction(self, complex_prediction): 
      if self.normalize:
        complex_prediction = self.reverse_normalize_fft(complex_prediction)
      ifft = tf.signal.ifft(complex_prediction)
      pred_audio = tf.cast(ifft, dtype=tf.float32)
      # pred_audio = self.overlap_add(pred_audio)
      return pred_audio

    def frames_to_audio(self, frames):
      audio = tf.signal.overlap_and_add(frames, self.hop)
      return audio

    # generate CNN input from audio frame
    def frames_to_fft_tensor(self, frames, window=True, center_fft=True): 
      if frames.ndim > 2:
        frames = tf.squeeze(frames)
      if window:
        frames *= self.win
      frames = tf.cast(frames, dtype=tf.complex64)
      # cut mirror
      fft = tf.signal.fft(frames)[:, :frames.shape[1]//2]
      fft = tf.expand_dims(fft, axis=2)
      fft_tensor = self.complex_to_ri(fft)
      if center_fft:
        fft_tensor = self.center_data(fft_tensor)
      return fft_tensor

    # generate audio frames from network predictions
    def fft_tensor_to_frames(self, fft_tensor, decenter_fft=True): 
      if decenter_fft:
        fft_tensor = self.de_center_data(fft_tensor)
      complex_tensor = self.ri_to_complex(fft_tensor)
      ifft = tf.signal.ifft(complex_tensor)
      frames = tf.cast(ifft, dtype=tf.float32)
      return frames

    # x, y input = audio frames, combines result
    def concat_xy_audio(self, x, y):
      assert x.ndim == 2, f"expecting 2 dimensions [batch size, frame size], recieved {x.shape}"
      x = tf.expand_dims(x, axis=1)
      y = tf.expand_dims(y, axis=1)
      concat = tf.concat([x, y], axis=1)
      concat = self.overlap_add(concat)
      return concat

    # predict audio frames at [self.y_offset] ahead of input_frames
    def predict_audio(self, input_frames, model, window=False): 
      assert input_frames.shape[-1] == self.block_size, f"input tensor {input_frames.shape} shape[-1] != {self.block_size}"
      if input_frames.ndim < 2: # in the case of a single audio frame
        input_frames = tf.expand_dims(input_frames, axis=0)
      if window:
        input_frames *= tf.signal.hann_window(input_frames.shape[1])
      model_input = self.frames_to_fft_tensor(input_frames)
      predicted_fft = model.predict(model_input)
      predicted_frames = self.fft_tensor_to_frames(predicted_fft)
      if predicted_frames.shape[0] == 1:
        predicted_frames = tf.squeeze(predicted_frames)
      return predicted_frames

    # center fft by interlacing freqs and concatenating mirror
    # this may improve training, with more information density towards the center of the vector,
    # and not to the sides, where convolution artifacts occur, and network density reduces
    # another goal is to achieve greater gaussian distribution by interleaving frequencies
    # in the network during the split/mirror process
    def center_data(self, fft_tensor):
      left = fft_tensor[:, ::2, :]
      right = fft_tensor[:, 1::2, :]
      left = tf.reverse(left, axis=[-2])
      centered_fft = tf.concat([left, right], axis=1)
      return centered_fft

    # reverse process of center_data()
    # un-mirrors and de-interlaces fft_tensors
    def de_center_data(self, fft_tensor):
      de_interlaced = np.zeros_like(fft_tensor)
      left = fft_tensor[:, :fft_tensor.shape[1]//2, :]
      right = fft_tensor[:, fft_tensor.shape[1]//2:, :]
      left = tf.reverse(left, axis=[-2])
      de_interlaced[:, ::2, :] = left
      de_interlaced[:, 1::2, :] = right
      return de_interlaced

    # return index to beginning of dataset
    def reset_generator(self):
      self.dataset_index = 0

    # main generator function for training
    def generate(self):
      self.dataset_index = 0
      while True:
        if self.dataset_index+self.batch_size > self.num_examples:
          self.dataset_index = 0

        audio_batch = self.data[self.dataset_index:self.dataset_index+self.batch_size]
        x_frames = np.zeros([self.batch_size, self.block_size])
        y_frames = np.zeros([self.batch_size, self.block_size])

        for i, a in enumerate(audio_batch):
          rand_idx = np.random.randint(0, high=len(a)-1-self.block_size-self.hop)
          x_frames[i, :] = a[rand_idx : rand_idx + self.block_size]
          y_frames[i, :] = a[rand_idx + self.hop : rand_idx + self.hop + self.block_size]

        x = self.frames_to_fft_tensor(x_frames, window=True, center_fft=True)
        y = self.frames_to_fft_tensor(y_frames, window=True, center_fft=True)
        self.dataset_index += self.batch_size
        yield x, y

class Generators():
    def __init__(self, dataset, batch_size, block_size, hop_ratio, offset):
        self.train_DG, self.val_DG = self.create_generators(
            dataset, 
            batch_size, 
            block_size, 
            hop_ratio, 
            offset)
        
    def create_generators(self, dataset, batch_size, block_size, hop_ratio, offset):
        train_DG = DataGenerator(batch_size, block_size, 2, dataset.train_data, hop_ratio, offset)
        val_DG = DataGenerator(batch_size, block_size, 2, dataset.val_data, hop_ratio, offset)
        return train_DG, val_DG