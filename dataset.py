# ----- NEXTBLOCK DATASET PARSING TOOLS -----

# ============================================================================
# Tools for loading dataset "chunks" of my custom Cambridge Multitrack dataset
# This dataset contains mismatched size (ragged) tensors of timbrally 
# verified PCM data. Windows of evenly sized tensors can be extracted from
# these snippets, which provides a much greater variety of data for training,
# and also more closely represents real-world scenarios of novel phase
# information, a key consideration in the design of this experiment.
# Currently the design is a bit messy, but I intend to clean everything up
# ============================================================================

import numpy as np
import resampy
import glob
import random
# import IPython.display as ipd
import matplotlib.pyplot as plt

class CambridgeDataset():

  def __init__(self, 
              chunk_path,
              train_val_split=0.8, 
              resamp=False, 
              sr_orig=44100,
              sr_new=16000):

    chunks = self.load_data_chunks(
      chunk_path, shuffle=True)

    self.train_data, self.val_data = self.dataset_from_chunks(
      chunks, train_val_split, resamp, sr_orig, sr_new)

  # Load all chunks, or set a limit if impatient or short on RAM
  def load_data_chunks(self, path, shuffle=True, limit=0):
    files = glob.glob(path+"/*.npy")
    assert(len(files)>0)
    print(f"num chunks found: {len(files)}")

    if shuffle:
      random.shuffle(files)
      
    chunks = []

    if limit < 1:
      limit = len(files)-1
    for filename in files[:limit]:
        print(f"loading {filename}")
        this_chunk = np.load(filename, allow_pickle=True)
        for c in this_chunk:
          chunks.append(c)
    return chunks

  # create numpy datasets (TF version in the works)
  def dataset_from_chunks(self, data_chunks, split=0.8, resamp=False, 
                    sr_orig=44100, sr_new=16000):
    samp_count = []
    for a in data_chunks:
      samps = 0
      for b in a:
        samps += b.shape[0]
      samp_count.append(samps)
    train_data = []
    val_data = []
    total_samps = np.sum(samp_count)
    print(f'total num samples {total_samps}')
    current_samp_count = 0
    fill_train = True
    i = 0
    for chunk, samps in zip(data_chunks, samp_count):
      if i % 100==0:
        print(f'chunk {i} of {len(data_chunks)}')
      current_samp_count += samps

      if current_samp_count >= int(total_samps*split):
        fill_train = False
      for c in chunk:
        if resamp:
          c = resampy.resample(c, sr_orig, sr_new)

        if fill_train:
          train_data.append(c)
        else:
          val_data.append(c)
      i+=1
    train_data = np.asarray(train_data)
    val_data = np.asarray(val_data)
    return train_data, val_data

  # Display random example in the dataset
  def VisualizeRandomExample(self, data, sr=44100):
    rnd_idx = np.random.randint(data.shape[0])
    plt.plot(data[rnd_idx])
    plt.show()
    # ipd.display(ipd.Audio(data[rnd_idx], rate=sr, autoplay=False))



  # slice up np dataset at random indicies to fit the window size
  # not used in main implementation
  def slice_np_dataset(self, data, window_size):
    output_arr = []
    for d in data:
      idx = np.random.randint(0,len(d)-window_size-1)
      output_arr.append(d[idx:idx+window_size])
    return np.asarray(output_arr)

# Class MemmapCambridgeDataset():
