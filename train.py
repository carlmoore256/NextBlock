from data_gen import DataGenerator, Generators
from models.nb_model import NB_Model
from dataset import CambridgeDataset
from callback import NB_Callback

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
dataset = CambridgeDataset(chunk_path="./Dataset", 
                            chunk_limit=10,
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
                    chkpoint_dir='./checkpoints/')

################## Training Callbacks ####################

cb = NB_Callback(generators.val_DG,
                nb_model,
                num_tests=1,
                test_interval=10,
                save_audio=True)

################## Fit Model ####################

nb_model.fit(generators, num_epochs, callbacks=[cb])