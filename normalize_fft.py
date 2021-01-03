import tensorflow as tf
import numpy as np
#testing how to normalize fft
# idea - multiply bins against a fletcher munson curve
# length = 256
# noise = np.random.uniform(-1.0,1.0,size=(length,1))

# freq=5
# sig = np.linspace(0,np.pi*2*freq,num=1024)
# sig = np.sin(sig)

length = 512
divisor = 4

rnd_idx = np.random.randint(train_data.shape[0])
rnd_samp = train_data[rnd_idx]
rnd_idx = np.random.randint(rnd_samp.shape[0])

sig = rnd_samp[rnd_idx:rnd_idx+length]

fft = tf.signal.fft(tf.cast(sig, tf.complex64))
fft = fft[:fft.shape[0]//2]

# very interesting - the signal divided by the fft equals a windowed version of the waveform - ask brent
# norm1 = sig/fft
scalar = 1.0/(length//divisor)

norm2 = tf.math.multiply(fft, scalar)

ifft = tf.signal.ifft(norm2*(length*(1/divisor)))
ifft = tf.math.real(ifft)

plt.plot(tf.math.real(fft))
plt.plot(tf.math.imag(fft))
plt.title('fft')
plt.show()
plt.plot(tf.math.real(norm2))
plt.plot(tf.math.imag(norm2))
plt.title('normalized fft')
plt.show()
plt.plot(ifft)
plt.title('ifft')
plt.plot(sig)
plt.title('original signal')
plt.plot()
