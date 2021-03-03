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

import matplotlib.pyplot as plt
import numpy as np
import resampy
import random
import glob
import os

class CambridgeDataset():

  def __init__(self, 
              dataset_path,
              train_val_split=0.8, 
              resamp=False, 
              sr_orig=44100,
              sr_new=16000):

    self.map_train = None
    self.map_val = None

    # automatically selects for memmap of npy directories          
    memmap_files = self.get_directory_files(dataset_path, "memmap")
    npy_files = self.get_directory_files(dataset_path, "npy")

    print(f'\n memmap files found {memmap_files}')
    print(f'npy files found {npy_files}\n')

    # loads the cambridge dataset from a memmap (preferred method)
    if len(memmap_files) > 0:
      # eventually implement better way to do this
      train_mmap_file = self.filter_file_list(memmap_files, "data_train")
      val_mmap_file = self.filter_file_list(memmap_files, "data_val")

      print(f'using {train_mmap_file} for train data')
      print(f'using {val_mmap_file} for validation data\n')

      self.train_data = np.memmap(train_mmap_file, dtype="float32", mode="r")
      self.val_data = np.memmap(val_mmap_file, dtype="float32", mode="r")

      print(f'training on {((len(self.train_data)/sr_orig)/60)/60} hrs of audio')
      print(f'validating on {((len(self.val_data)/sr_orig)/60)/60} hrs of audio\n')
      # get the map to the memmap which provides indicies
      self.map_train = np.load(self.filter_file_list(npy_files, "map_train"))
      self.map_val = np.load(self.filter_file_list(npy_files, "map_val"))

      print(f'train memmap examples {self.map_train.shape}')
      print(f'val memmap examples {self.map_val.shape}\n')

    else: # if we're loading chunks (deprecated)
      chunks = self.load_data_chunks(
        dataset_path, shuffle_chunks=True)

      self.train_data, self.val_data = self.dataset_from_chunks(
        chunks, train_val_split, resamp, sr_orig, sr_new)

  # return the first closest match to filter from a list of files
  def filter_file_list(self, files, search_term):
    filtered = filter(lambda a: search_term in a, files)
    ffl = list(filtered)
    assert len(ffl) > 0, f'failed loading dataset, no matches in {files} ending with {search_term}'
    if len(ffl) > 1:
      print(f'found more than 1 file for {ffl}, choosing {ffl[0]} \n \
            only one memmap per split supported')
    return ffl[0]

  # returns all files of type extension in the given directory
  def get_directory_files(self, path, extension):
    return glob.glob(f"{path}/*.{extension}")

  # Load all chunks, or set a limit if impatient or short on RAM
  def load_data_chunks(self, path, shuffle_chunks=True, limit=0):
    files = self.get_directory_files(path, "npy")
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
