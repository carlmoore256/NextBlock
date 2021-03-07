import tensorflow.keras.callbacks as tf_cb
from data_gen import DataGenerator, Generators
from models.nb_model import NB_Model
from dataset import CambridgeDataset
from callback import NB_Callback
from datetime import datetime
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("-dataset", type=str, default='E:/Datasets/VoxVerified', help="path to dataset")
parser.add_argument("-model", type=str, default='', help="path to existing model to load")
parser.add_argument("-b", type=int, default=32, help="batch size")
parser.add_argument("-k", type=int, default=9, help="kernel size")
parser.add_argument("-e", type=int, default=100, help="epochs")
parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-sr", type=int, default=44100, help="samplerate")

parser.add_argument("-blocksize", type=int, default=256, help="audio block size")
parser.add_argument("-hopratio", type=int, default=2, help="hop size as ratio, hop in samples = block size / hop")
parser.add_argument("-predictoffset", type=int, default=1, help="number of hops ahead for model to predict during training")
parser.add_argument("-normalize", type=bool, default=True, help="normalize fft to 1/fftsize")

# model creation parameters - defaults determined by some global optimization done a while back, but by no means the right answer
parser.add_argument("--iofilt", type=int, default=32, help="number of filters on outer-most layers")
parser.add_argument("--bottleneck", type=int, default=8, help="size of latent dimension")
parser.add_argument("--bias", type=bool, default=True, help="use bias")

args = parser.parse_args()



load_model = args.model
dataset_path = args.dataset

############### Hyper Parameters ####################

batch_size = args.b
num_epochs = args.e
learning_rate = args.lr
block_size = args.blocksize

############## Training Parameters ##################

hop_ratio = args.hopratio
kernel_size = args.k

# how far ahead to predict a block of audio, given input X
# value is expressed in how many hops ahead to predict:
# offset = (block_size/hop_ratio) * y_offset
prediction_offset = args.predictoffset

# this model uses a full samplerate for real-time applications
# in which downsampling/upsampling latency is impractical
sr = args.sr

################### Dataset ####################

# refer to CambridgeDataset for more information
dataset = CambridgeDataset(dataset_path=dataset_path, 
                            train_val_split=0.8,
                            resamp=False)

################## Generators ####################

# holds training and validation generator classes
generators = Generators(dataset, 
                        batch_size, 
                        block_size, 
                        hop_ratio,
                        prediction_offset,
                        args.normalize)

################## Build Model ####################

# network input will be block_size/2, with 2 channels
# for phase and magnitude: [..., block_size/2, 2]
nb_model = NB_Model(input_shape=[block_size//2,2],
                    learning_rate=learning_rate, 
                    loss='mse',
                    model_dir='./saved/',
                    chkpoint_dir='/tmp/checkpoint/')

# create a u-net model for making inference on fft training data

if load_model == '':
  nb_model.create_unet_fft(
                      lr=learning_rate, 
                      filters=args.iofilt, # num filters on input & output layers
                      kernel_size=kernel_size, 
                      bottleneck=args.bottleneck, # size of the network's latent dimension
                      use_bias=args.bias, 
                      strides=2, 
                      activation='tanh')
else:
  nb_model.load(load_model, learning_rate)

################## Training Callbacks ####################

cb = NB_Callback(generators.val_DG,
                nb_model,
                num_tests=1,
                test_interval=10,
                save_audio=False)

################## Fit Model ####################

now = datetime.now()
logdir = "./tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
os.mkdir(logdir)

nb_model.fit(generators, 
            num_epochs, 
            callbacks=[
                      cb, 
                      # tf_cb.ModelCheckpoint(filepath='E:\Datasets\VoxVerified\checkpoints\model.{epoch:02d}-{val_loss:.2f}.h5',save_freq='epoch'),
                      tf_cb.TensorBoard(log_dir=logdir)])

nb_model.save()