from tensorflow.keras.callbacks import Callback
from tensorflow.audio import encode_wav
from tensorflow.io import write_file
from tensorflow import expand_dims
import matplotlib.pyplot as plt
from threading import Thread
from pathlib import Path
import numpy as np

class NB_Callback(Callback):
    def __init__(self, 
                dataGen, 
                nb_model, 
                num_tests=1,
                test_interval=10,
                save_audio=True,
                samplerate=44100,
                save_audio_dir='./audio/training_clips'):

        self.losses = []
        self.model = nb_model.model
        self.dataGen = dataGen
        assert num_tests <= dataGen.batch_size, "num_tests must be <= batch_size"
        self.num_tests = num_tests
        self.samplerate = samplerate
        self.save_audio = save_audio
        self.save_audio_dir = save_audio_dir
        if save_audio:
            Path(save_audio_dir).mkdir(parents=True, exist_ok=True)

        self.test_interval = test_interval
        self.interval_count = 0
        
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        if self.interval_count % self.test_interval == 0:
            self.plot_async()
        #   plots = Thread(target=self.plot_async, daemon=True)
        self.interval_count += 1
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return

    def save_audio_clip(self, samples, path, name):
        samples = expand_dims(samples, -1)
        audio = encode_wav(samples, self.samplerate)
        write_file(f'{path}/{name}', audio)

    # plots current results on-screen asyncronously, allowing training to continue in bg
    def plot_async(self):
        x, y = next(self.dataGen.generate())
        y_p = self.model.predict(x)
        x_frames = self.dataGen.fft_tensor_to_frames(x, decenter_fft=True)
        y_frames = self.dataGen.fft_tensor_to_frames(y, decenter_fft=True)
        yp_frames = self.dataGen.fft_tensor_to_frames(y_p, decenter_fft=True)

        for i in range(self.num_tests):

            x_yp = np.pad(x_frames[i,:], [0, self.dataGen.hop]) + np.pad(yp_frames[i,:], [self.dataGen.hop, 0])

            plt.title('x + predicted y', color='white')
            plt.plot(x_yp, color='black')
            plt.plot(np.pad(x_frames[i,:], [0, self.dataGen.hop]))
            plt.plot(np.pad(yp_frames[i,:], [self.dataGen.hop, 0]))
            plt.show()

            plt.title('ground truth fft')
            plt.plot(y[i,:,:])
            plt.show()

            plt.title('predicted fft')
            plt.plot(y_p[i, :, :])
            plt.show()

            if self.save_audio:
                print(x_frames.shape)
                print(y_frames.shape)
                print(yp_frames.shape)
                print(x_yp.shape)
                self.save_audio_clip(x_frames[i,:], self.save_audio_dir, f"e{epoch}_{i}_input_samp_.wav")
                self.save_audio_clip(y_frames[i,:], self.save_audio_dir, f"e{epoch}_{i}_ground_truth.wav")
                self.save_audio_clip(yp_frames[i,:], self.save_audio_dir, f"e{epoch}_{i}_predicted.wav")
                self.save_audio_clip(x_yp, self.save_audio_dir, f"e{epoch}_{i}_input_pred_combined.wav")

        plt.title('loss', color='white')
        plt.plot(np.asarray(self.losses))
        plt.show()