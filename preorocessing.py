import tensorflow as tf


def load_audio(dir):
    # tensorflow read file (can read any file)  
    raw_audio = tf.io.read_file(dir)
    # decodes raw_audio into -1f to 1f float32 tensor
    waveform = tf.audio.decode_wav(raw_audio, desired_channels=1)
    # waveform[0]: float32 tensor of waveform data
    # waveform[1]: samplerate (ignoring sr for now)
    return waveform[0]

# waveform audio -> FFT (tf.complex64 dtype)
def fft(audio, normalize):
    fft = tf.signal.fft(audio)
    if normalize:
        fft = normalize_fft(fft)
    return fft

def normalize_fft(fft):
    scalar = 1.0/self.block_size
    normalized_fft = tf.math.multiply(fft, scalar)
    return normalized_fft

def reverse_normalize_fft(normalized_fft):
    return normalized_fft * self.block_size

# x + y(i) -> magnitude, angle
def rectangular_to_polar(rectangular):
    magnitude = tf.abs(rectangular)
    angle = tf.math.angle(rectangular)
    polar = tf.concat([magnitude, angle], axis=2)
    return polar

# loads a single file as an FFT tensor to be fed into the net
def load_single(file, win):
    audio = load_audio(file)
    print(f'audio shape {audio.shape}')
    frames = tf.squeeze(tf.signal.frame(audio, self.block_size, self.hop, axis=0))
    frames *= win
    tensor = frames_to_fft_tensor(frames, window=window)
    return audio, frames, tensor

# float32 tensor to rectangular notation:
# [real, imaginary] -> [complex,]
def complex_to_ri(tensor):      
    real = tf.math.real(tensor)
    imag = tf.math.imag(tensor)
    ri_t = tf.concat([real, imag], axis=2)
    return ri_t

# rectangular notation to float32 tensor
# [complex,] -> [real, imaginary]
def ri_to_complex(tensor):      
    real = tensor[:,:,0]
    imag = tensor[:,:,1]
    # account for FFT mirror cutoff at N/2+1
    mirror = tf.zeros_like(real)
    real = tf.concat([real,mirror], axis=1)
    imag = tf.concat([imag,mirror], axis=1)
    complex_t = tf.dtypes.complex(real, imag)
    return complex_t

def overlap_add(frames):
    audio = tf.signal.overlap_and_add(frames, hop)
    return audio

# prediction in complex notation -> audio tensor
def ifft_prediction(complex_prediction, normalize=True): 
    if normalize:
        complex_prediction = reverse_normalize_fft(complex_prediction)
    ifft = tf.signal.ifft(complex_prediction)
    pred_audio = tf.cast(ifft, dtype=tf.float32)
    # pred_audio = self.overlap_add(pred_audio)
    return pred_audio

def frames_to_audio(frames, hop):
    audio = tf.signal.overlap_and_add(frames, hop)
    return audio

# generate CNN input from audio frame
def frames_to_fft_tensor(frames, win, center_fft=True): 
    if frames.ndim > 2:
        frames = tf.squeeze(frames)
    frames *= win
    frames = tf.cast(frames, dtype=tf.complex64)
    # cut mirror
    fft = tf.signal.fft(frames)[:, :frames.shape[1]//2]
    fft = tf.expand_dims(fft, axis=2)
    fft_tensor = complex_to_ri(fft)
    if center_fft:
        fft_tensor = center_data(fft_tensor)
    return fft_tensor

# generate audio frames from network predictions
def fft_tensor_to_frames( fft_tensor, decenter_fft=True): 
    if decenter_fft:
        fft_tensor = de_center_data(fft_tensor)
    complex_tensor = ri_to_complex(fft_tensor)
    ifft = tf.signal.ifft(complex_tensor)
    frames = tf.cast(ifft, dtype=tf.float32)
    return frames

# x, y input = audio frames, combines result
def concat_xy_audio(x, y):
    assert x.ndim == 2, f"expecting 2 dimensions [batch size, frame size], recieved {x.shape}"
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    concat = tf.concat([x, y], axis=1)
    concat = overlap_add(concat)
    return concat