#coding=utf-8
import os
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
from random import shuffle
import speechpy
import datetime
import tables

######################################
####### Define the dataset class #####
######################################
class AudioDataset():
    """Audio dataset."""

    def __init__(self, files_path, audio_dir, transform=None):
        """
        Args:
            files_path (string): Path to the .txt file which the address of files are saved in it.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # self.sound_files = [x.strip() for x in content]
        self.audio_dir = audio_dir
        self.transform = transform

        # Open the .txt file and create a list from each line.
        with open(files_path, 'r') as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        list_files = []
        for x in content:
            sound_file_path = os.path.join(self.audio_dir, x.strip().split()[1])
            # print('file path is '+'*'*10)
            # print(sound_file_path)
            try:
                with open(sound_file_path, 'rb') as f:
                    riff_size, _ = wav._read_riff_chunk(f)
                    file_size = os.path.getsize(sound_file_path)

                # Assertion error.
                assert riff_size == file_size and os.path.getsize(sound_file_path) > 1000, "Bad file!"

                # Add to list if file is OK!
                list_files.append(x.strip())
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print('file %s is corrupted!' % sound_file_path)
            # except:
            #     print("Unexpected error:", sys.exc_info()[0])
            #     raise

        # Save the correct and healthy sound files to a list.
        self.sound_files = list_files

    def __len__(self):
        return len(self.sound_files)

    def __getitem__(self, idx):
        # Get the sound file path
        sound_file_path = os.path.join(self.audio_dir, self.sound_files[idx].split()[1])
        # print(sound_file_path)
        ##############################
        ### Reading and processing ###
        ##############################

        # Reading .wav file
        fs, signal = wav.read(sound_file_path)

        # Reading .wav file
        import soundfile as sf
        signal, fs = sf.read(sound_file_path)

        ###########################
        ### Feature Extraction ####
        ###########################

        # DEFAULTS:
        num_coefficient = 40

        # Staching frames
        # frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025,
        #                                           frame_stride=0.01,
        #                                           zero_padding=True)

        # # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        # power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]

        logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
                                          num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                          high_frequency=None)

        ########################
        ### Handling sample ####
        ########################

        # Label extraction
        # print(self.sound_files)
        label = int(self.sound_files[idx].split()[0][1:])

        sample = {'feature': logenergy, 'label': label}

        ##################import os



        if self.transform:

            sample = self.transform(sample)
        else:
            feature, label = sample['feature'], sample['label']
            sample = feature, label

        return sample
        # return sample


class CMVN(object):
    """Cepstral mean variance normalization.

    """

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        # Mean variance normalization of the spectrum.
        # The following line should be Uncommented if cepstral mean variance normalization is desired!
        # feature = speechpy.processing.cmvn(feature, variance_normalization=True)

        return {'feature': feature, 'label': label}

class Feature_Cube(object):
    """Return a feature cube of desired size.

    Args:
        cube_shape (tuple): The shape of the feature cube.
    """

    def __init__(self, cube_shape, augmentation=True):
        assert isinstance(cube_shape, (tuple))
        self.augmentation = augmentation
        self.cube_shape = cube_shape
        self.num_utterances = cube_shape[0]
        self.num_frames = cube_shape[1]
        self.num_coefficient = cube_shape[2]


    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        # Feature cube.
        feature_cube = np.zeros((self.num_utterances, self.num_frames, self.num_coefficient), dtype=np.float32)

        if self.augmentation:
            # Get some random starting point for creation of the future cube of size (num_frames x num_coefficient x num_utterances)
            # Since we are doing random indexing, the data augmentation is done as well because in each iteration it returns another indexing!
            idx = np.random.randint(feature.shape[0] - self.num_frames, size=self.num_utterances)
            for num, index in enumerate(idx):
                feature_cube[num, :, :] = feature[index:index + self.num_frames, :]
        else:
            idx = range(self.num_utterances)
            for num, index in enumerate(idx):
                feature_cube[num, :, :] = feature[index:index + self.num_frames, :]



        # return {'feature': feature_cube, 'label': label}
        return {'feature': feature_cube[None, :, :, :], 'label': label}


class ToOutput(object):
    """Return the output.

    """

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        feature, label = sample['feature'], sample['label']
        return feature, label

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> Compose([
        >>>     CMVN(),
        >>>     Feature_Cube(cube_shape=(20, 80, 40),
        >>>     augmentation=True), ToOutput(),
        >>>        ])
        If necessary, for the details of this class, please refer to Pytorch documentation.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
def get_batch(max_index=300,num_utterances=20,lenght=10,file_path=None):
    max_index=len(open(file_path).readlines())
    audio_dir='/home/x/Document/tongdun_data/data_aishell/wav/train'
    data=np.zeros((lenght,80,40,num_utterances))
    dataset = AudioDataset(files_path=file_path, audio_dir=audio_dir,
                           transform=Compose([CMVN(), Feature_Cube(cube_shape=(1, 80, 40), augmentation=True), ToOutput()]))
    label=[int(file_path[-8:-4])]*lenght
    for i in range(lenght):
        indexs=np.random.randint(1,max_index,num_utterances)
        batch_features=[dataset.__getitem__(idx)[0][0,:,:,:] for idx in indexs]
        print(batch_features[0].shape)
        batch_features=np.vstack(batch_features)
        batch_features=np.swapaxes(batch_features,0,1)
        batch_features=np.swapaxes(batch_features,1,2)
        print(batch_features.shape)
        data[i,:,:,:]=batch_features
    label=np.array(label)

    return data,label
    
    


if __name__ == '__main__':
    
    train_path_root='/home/x/Document/tongdun_data/data_aishell/wav/train'
    subjects=os.listdir(train_path_root)[:2]

    # subjects=['S0099', 'S0052', 'S0055', 'S0251', 'S0063', 'S0097', 'S0090', 'S0064', 'S0666', 'S0202', 'S0030', 'S0659', 'S0037', 'S0205']
    all_feature_train,all_label_train=[],[]
    all_feature_test,all_label_test=[],[]
    for subject in subjects:
        file_path='/home/x/Document/tongdun_data/data_aishell/wav/path_map/train/{}.txt'.format(subject)

        feature_train,label_train=get_batch(lenght=100,file_path=file_path)
        feature_test,label_test=get_batch(lenght=2,file_path=file_path)

        all_feature_train.append(feature_train)
        all_label_train.append(label_train)
        all_feature_test.append(feature_test)
        all_label_test.append(label_test)

    h5file = tables.open_file("train.h5", mode="w")
    label_test,label_train=np.hstack(all_label_test),np.hstack(all_label_train)
    utterance_test,utterance_train=np.vstack(all_feature_test),np.vstack(all_feature_train)
    h5file.create_array(h5file.root, 'label_test',label_test)
    h5file.create_array(h5file.root, 'label_train',label_train)
    h5file.create_array(h5file.root, 'utterance_test',utterance_test)
    h5file.create_array(h5file.root, 'utterance_train',utterance_train)
    print(h5file)
    