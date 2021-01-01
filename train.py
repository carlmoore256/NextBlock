from data_gen import DataGenerator, Generators
from models.nb_model import NB_Model
from dataset import CambridgeDataset
from callback import NB_Callback

load_model = '/content/saved'

############### Hyper Parameters ####################

batch_size = 32
num_epochs = 100
learning_rate = 1e-4
block_size = 512

############## Training Parameters ##################

hop_ratio = 2
kernel_size = 9

# how far ahead to predict a block of audio, given input X
# value is expressed in how many hops ahead to predict:
# offset = (block_size/hop_ratio) * y_offset
prediction_offset = 1

# this model uses a full samplerate for real-time applications
# in which downsampling/upsampling latency is impractical
sr = 44100

################### Dataset ####################

# refer to CambridgeDataset for more information
dataset = CambridgeDataset(chunk_path="/content/drive/My Drive/Datasets/VoxVerified/", 
                            chunk_limit=0, # REMOVE LIMIT if you want to load everything
                            train_val_split=0.8,
                            resamp=False)

################## Generators ####################

# holds training and validation generator classes
generators = Generators(dataset, 
                        batch_size, 
                        block_size, 
                        hop_ratio, 
                        prediction_offset)

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
                      filters=512, # num filters on input & output layers
                      kernel_size=kernel_size, 
                      bottleneck=8, # size of the network's latent dimension
                      use_bias=False, 
                      strides=2, 
                      activation='tanh')
else:
  nb_model.load(load_model, learning_rate)

################## Training Callbacks ####################

cb = NB_Callback(generators.val_DG,
                nb_model,
                num_tests=1,
                test_interval=10,
                save_audio=True)

################## Fit Model ####################

nb_model.fit(generators, num_epochs, callbacks=[cb])

nb_model.save()