import torch
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import Dataset

import numpy as np
import random
import pandas as pd
import csv
import argparse
from tqdm import tqdm
import librosa
import json
import os


def make_index_dict(label_csv):
	index_lookup = {}
	with open(label_csv, 'r') as f:
		csv_reader = csv.DictReader(f)
		for row in csv_reader:
			index_lookup[row['mids']] = row['index']
	return index_lookup


class FSD50K(Dataset):
	
	def __init__(self, cfg, split='train', transform=None, norm_stats=None, crop_frames=None):
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.split = split
		self.transform = transform
		self.norm_stats = norm_stats
		self.crop_frames = self.cfg.crop_frames if crop_frames is None else crop_frames

		self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
		self.to_melspecgram = AT.MelSpectrogram(
			sample_rate=cfg.sample_rate,
			n_fft=cfg.n_fft,
			win_length=cfg.win_length,
			hop_length=cfg.hop_length,
			n_mels=cfg.n_mels,
			f_min=cfg.f_min,
			f_max=cfg.f_max,
			power=2,
		)
		# load in csv files
		if split != 'test':
			self.df = pd.read_csv("/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.ground_truth/dev.csv", header=None)
			if split == 'train_val':
				pass 
			elif split == 'train':
				self.df = self.df[self.df.iloc[:, 3] == 'train']
			elif split == 'val':
				self.df = self.df[self.df.iloc[:, 3] == 'val']
		else:
			self.df = pd.read_csv("/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.ground_truth/eval.csv", header=None)	
		self.files = np.asarray(self.df.iloc[:, 0], dtype=str)
		self.labels = np.asarray(self.df.iloc[:, 2], dtype=str)  # mids (separated by ,)
		self.index_dict = make_index_dict("/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.ground_truth/vocabulary.csv")
		self.label_num = len(self.index_dict)


	def __len__(self):
		return len(self.files)
		
		
	def __getitem__(self, idx):
		fname = self.files[idx]
		labels = self.labels[idx]
		# initialize the label
		label_indices = np.zeros(self.label_num)
		# add sample labels
		for label_str in labels.split(','):
			label_indices[int(self.index_dict[label_str])] = 1.0
		label_indices = torch.FloatTensor(label_indices)
		if self.cfg.load_lms:
			# load lms
			if self.split != 'test':
				audio_path = "/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.dev_audio/" + fname + ".npy"
			else:
				audio_path = "/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.eval_audio/" + fname + ".npy"
			lms = torch.tensor(np.load(audio_path)).unsqueeze(0)
			# Trim or pad
			l = lms.shape[-1]
			if l > self.crop_frames:
				start = np.random.randint(l - self.crop_frames)
				lms = lms[..., start:start + self.crop_frames]
			elif l < self.crop_frames:
				pad_param = []
				for i in range(len(lms.shape)):
					pad_param += [0, self.crop_frames - l] if i == 0 else [0, 0]
				lms = F.pad(lms, pad_param, mode='constant', value=0)
			lms = lms.to(torch.float)
		else:
			# load raw audio
			if self.split != 'test':
				audio_path = "/vol/bitbucket/jla21/proj/data/FSD50K/FSD50K.dev_audio/" + fname + ".wav"
			else:
				audio_path = "/vol/bitbucket/jla21/proj/data/FSD50K/FSD50K.eval_audio/" + fname + ".wav"
			wav, org_sr = librosa.load(audio_path, sr=self.cfg.sample_rate)
			wav = torch.tensor(wav)  # (length,)
			# zero padding to both ends
			length_adj = self.unit_length - len(wav)
			if length_adj > 0:
				half_adj = length_adj // 2
				wav = F.pad(wav, (half_adj, length_adj - half_adj))
			# random crop unit length wave
			length_adj = len(wav) - self.unit_length
			start = random.randint(0, length_adj) if length_adj > 0 else 0
			wav = wav[start:start + self.unit_length]
			# to log mel spectogram -> (1, n_mels, time)
			lms = (self.to_melspecgram(wav) + torch.finfo().eps).log()
			lms = lms.unsqueeze(0)
		# normalise lms with pre-computed dataset statistics
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
		# transforms to lms
		if self.transform is not None:
			lms = self.transform(lms)

		return lms, label_indices


class LibriSpeech(Dataset):
	
	def __init__(self, cfg, train=True, transform=None, norm_stats=None, n_dummy=200):
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.train = train
		self.transform = transform
		self.norm_stats = norm_stats
		self.n_dummy = n_dummy
		if self.cfg.load_lms:
			self.base_path= "data/LibriSpeech_lms/"
		else:
			self.base_path = "data/LibriSpeech/"

		self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
		self.to_melspecgram = AT.MelSpectrogram(
			sample_rate=cfg.sample_rate,
			n_fft=cfg.n_fft,
			win_length=cfg.win_length,
			hop_length=cfg.hop_length,
			n_mels=cfg.n_mels,
			f_min=cfg.f_min,
			f_max=cfg.f_max,
			power=2,
		)
		# load in json file
		self.datapath = self.base_path + "librispeech_tr960_cut.json"
		with open(self.datapath, 'r') as fp:
			data_json = json.load(fp)
		self.data = data_json.get('data')
		

	def __len__(self):
		return len(self.data)
		
		
	def __getitem__(self, idx):
		datum = self.data[idx]
		fname = datum.get('wav')
		dummy_label = torch.zeros(self.n_dummy)

		if self.cfg.load_lms:
			# load lms
			audio_path = self.base_path + fname[:-len(".flac")] + ".npy"
			lms = torch.tensor(np.load(audio_path)).unsqueeze(0)
			# Trim or pad
			l = lms.shape[-1]
			if l > self.cfg.crop_frames:
				start = np.random.randint(l - self.cfg.crop_frames)
				lms = lms[..., start:start + self.cfg.crop_frames]
			elif l < self.cfg.crop_frames:
				pad_param = []
				for i in range(len(lms.shape)):
					pad_param += [0, self.cfg.crop_frames - l] if i == 0 else [0, 0]
				lms = F.pad(lms, pad_param, mode='constant', value=0)
			lms = lms.to(torch.float)
		else:
			# load raw audio
			audio_path = self.base_path + fname
			wav, org_sr = librosa.load(audio_path, sr=self.cfg.sample_rate)
			wav = torch.tensor(wav)  # (length,)
			# zero padding to both ends
			length_adj = self.unit_length - len(wav)
			if length_adj > 0:
				half_adj = length_adj // 2
				wav = F.pad(wav, (half_adj, length_adj - half_adj))
			# random crop unit length wave
			length_adj = len(wav) - self.unit_length
			start = random.randint(0, length_adj) if length_adj > 0 else 0
			wav = wav[start:start + self.unit_length]
			# to log mel spectogram -> (1, n_mels, time)
			lms = (self.to_melspecgram(wav) + torch.finfo().eps).log()
			lms = lms.unsqueeze(0)
		# normalise lms with pre-computed dataset statistics
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
		# transforms to lms
		if self.transform is not None:
			lms = self.transform(lms)

		return lms, dummy_label


class NSynth_HEAR(Dataset):
	
	def __init__(self, cfg, split='train', transform=None, norm_stats=None):
		super().__init__()
		
		# initializations
		self.cfg = cfg
		self.split = split
		self.transform = transform
		self.norm_stats = norm_stats
		self.base_path = "hear/tasks/nsynth_pitch-v2.2.3-50h/"
		self.data_path = self.base_path + f"16000/{split}/" 

		self.jsonpath = self.base_path + f"{split}.json"
		with open(self.jsonpath, 'r') as fp:
			data_json = json.load(fp)
		self.data = [(name, label[0]) for name, label in data_json.items()]

		self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
		self.to_melspecgram = AT.MelSpectrogram(
			sample_rate=cfg.sample_rate,
			n_fft=cfg.n_fft,
			win_length=cfg.win_length,
			hop_length=cfg.hop_length,
			n_mels=cfg.n_mels,
			f_min=cfg.f_min,
			f_max=cfg.f_max,
			power=2,
		)

		
	def __len__(self):
		return len(self.data)
		
		
	def __getitem__(self, idx):
		fname, label = self.data[idx]
		label = int(label - 21)  # convert pitch to index

		if self.cfg.load_lms:
			# load lms
			audio_path = f"data/nsynth_lms/nsynth-{self.split}/audio/{fname[:-len('.wav')]}.npy"
			lms = torch.tensor(np.load(audio_path)).unsqueeze(0)
			# Trim or pad
			l = lms.shape[-1]
			if l > self.cfg.crop_frames:
				start = np.random.randint(l - self.cfg.crop_frames)
				lms = lms[..., start:start + self.cfg.crop_frames]
			elif l < self.cfg.crop_frames:
				pad_param = []
				for i in range(len(lms.shape)):
					pad_param += [0, self.cfg.crop_frames - l] if i == 0 else [0, 0]
				lms = F.pad(lms, pad_param, mode='constant', value=0)
			lms = lms.to(torch.float)
		else:
			# load raw audio
			audio_path = self.data_path + fname
			wav, org_sr = librosa.load(audio_path, sr=self.cfg.sample_rate)
			wav = torch.tensor(wav)  # (length,)
			# zero padding to both ends
			length_adj = self.unit_length - len(wav)
			if length_adj > 0:
				half_adj = length_adj // 2
				wav = F.pad(wav, (half_adj, length_adj - half_adj))
			# random crop unit length wave
			length_adj = len(wav) - self.unit_length
			start = random.randint(0, length_adj) if length_adj > 0 else 0
			wav = wav[start:start + self.unit_length]
			# to log mel spectogram -> (1, n_mels, time)
			lms = (self.to_melspecgram(wav) + torch.finfo().eps).log()
			lms = lms.unsqueeze(0)
		# normalise lms with pre-computed dataset statistics
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
		# transforms to lms
		if self.transform is not None:
			lms = self.transform(lms)

		return lms, label

"""
class AudioSet(Dataset):
	def __init__(self, cfg, transform=None, norm_stats=None):
		super().__init__()

		self.cfg = cfg 
		self.transform = transform
		self.norm_stats = norm_stats
		self.base_dir = "/vol/bitbucket/jla21/proj/data/audioset_lms/"

		# load in csv file
		df = pd.read_csv(os.path.join(self.base_dir, "unbalanced_train_segments-downloaded.csv"), header=None)
		# first column contains the audio fnames
		self.audio_fnames = np.asarray(df.iloc[:, 0])
		# second column contains the labels (separated by # for multi-label)
		self.labels = np.asarray(df.iloc[:, 1])
		# third column contains the identifier (balanced_train_segments or unbalanced_train_segments)
		self.ident = np.asarray(df.iloc[:, 2])
		# load in class labels and create label -> index look-up dict 
		self.index_dict = make_index_dict(os.path.join(self.base_dir, "class_labels_indices.csv"))
		self.label_num = len(self.index_dict)

		# also read in FSD50K csv files (in case of ValueErrors for incorrectly downloaded AS samples)
		df_fsd50k = pd.read_csv("/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.ground_truth/dev.csv", header=None)
		self.files_fsd50k = np.asarray(df_fsd50k.iloc[:, 0], dtype=str)

	def __len__(self):
		return len(self.audio_fnames)


	def __getitem__(self, idx):
		
		audio_fname = self.audio_fnames[idx]
		labels = self.labels[idx]
		ident = self.ident[idx]
		# initialize the label
		label_indices = np.zeros(self.label_num)
		# add sample labels
		for label_str in labels.split('#'):
			label_indices[int(self.index_dict[label_str])] = 1.0
		label_indices = torch.FloatTensor(label_indices)
		# load .npy spectrograms 
		audio_fpath = os.path.join(os.path.join(*[self.base_dir, "unbalanced_train_segments", f"{audio_fname}.npy"]))
		try:
			lms = torch.tensor(np.load(audio_fpath)).unsqueeze(0)
		except ValueError:
			fname = np.random.choice(self.files_fsd50k)
			audio_fpath = "/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.dev_audio/" + fname + ".npy"
			lms = torch.tensor(np.load(audio_fpath)).unsqueeze(0)
		# Trim or pad
		l = lms.shape[-1]
		if l > self.cfg.crop_frames:
			start = np.random.randint(l - self.cfg.crop_frames)
			lms = lms[..., start:start + self.cfg.crop_frames]
		elif l < self.cfg.crop_frames:
			pad_param = []
			for i in range(len(lms.shape)):
				pad_param += [0, self.cfg.crop_frames - l] if i == 0 else [0, 0]
			lms = F.pad(lms, pad_param, mode='constant', value=0)
		lms = lms.to(torch.float)
		#print(f"the shape of spectrogram is:{lms.shape}")
		if self.norm_stats is not None:
			lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
		# transforms
		if self.transform is not None:
			lms = self.transform(lms)
			
		return lms, label_indices
"""
class AudioSet(Dataset):
	def __init__(self, cfg, transform=None, norm_stats=None):
		super().__init__()

		self.cfg = cfg 
		self.transform = transform
		self.norm_stats = norm_stats
		self.base_dir = "/vol/bitbucket/jla21/proj/data/audioset"

		# load in csv file
		df = pd.read_csv(os.path.join('/vol/bitbucket/zw1222/proj', "unbalanced_train_segments-downloaded.csv"), header=None)
		# first column contains the audio fnames
		self.audio_fnames = np.asarray(df.iloc[:, 0])
		# second column contains the labels (separated by # for multi-label)
		self.labels = np.asarray(df.iloc[:, 1])
		# third column contains the identifier (balanced_train_segments or unbalanced_train_segments)
		self.ident = np.asarray(df.iloc[:, 2])
		# load in class labels and create label -> index look-up dict 
		self.index_dict = make_index_dict(os.path.join(self.base_dir, "class_labels_indices.csv"))
		self.label_num = len(self.index_dict)

		self.temp_dir = '/vol/bitbucket/zw1222/proj/temp'
		self.to_melspecgram = AT.MelSpectrogram(
			sample_rate=16000,
			n_fft=1024,
			win_length=1024,
			hop_length=160,
			n_mels=64,
			f_min=60,
			f_max=7800,
			power=2,
		)
		# also read in FSD50K csv files (in case of ValueErrors for incorrectly downloaded AS samples)
		#df_fsd50k = pd.read_csv("/vol/bitbucket/jla21/proj/data/FSD50K_lms/FSD50K.ground_truth/dev.csv", header=None)
		#self.files_fsd50k = np.asarray(df_fsd50k.iloc[:, 0], dtype=str)

	def __len__(self):
		return len(self.audio_fnames)


	def __getitem__(self, idx):
		
		audio_fname = self.audio_fnames[idx]
		labels = self.labels[idx]
		ident = self.ident[idx]
		# initialize the label
		label_indices = np.zeros(self.label_num)
		# add sample labels
		for label_str in labels.split('#'):
			label_indices[int(self.index_dict[label_str])] = 1.0
		label_indices = torch.FloatTensor(label_indices)
		print(f"current piece of data is :{idx}")
		# load wav files:
		audio_fpath = os.path.join(os.path.join(*[self.base_dir, "unbalanced_train_segments", f"{audio_fname}.wav"]))
		if self.cfg.mp3_compression:

			bitrate_1 = '128k' #randomise
			bitrate_2 = '64k' #randomise
			wav_1, _ = extract_compressed_wav(audio_fpath, self.temp_dir, bitrate=bitrate_1)
			wav_2, _ = extract_compressed_wav(audio_fpath, self.temp_dir, bitrate=bitrate_2)
			lms_1 = (self.to_melspecgram(wav_1) + torch.finfo().eps).log().unsqueeze(0)
			lms_2 = (self.to_melspecgram(wav_2) + torch.finfo().eps).log().unsqueeze(0)
			lms_1, lms_2 = trim_pad(self.cfg, lms_1), trim_pad(self.cfg, lms_2)
			if self.norm_stats is not None:
				lms_1 = (lms_1- self.norm_stats[0]) / self.norm_stats[1]
				lms_2 = (lms_2- self.norm_stats[0]) / self.norm_stats[1]
			#transforms (multitransform is false fo mp3/ldm compression)
			if self.transform is not None:
				lms = [self.transform(lms_1), self.transform(lms_2)]
			else:
				lms = [lms_1, lms_2]
			return lms, label_indices
			
		elif self.cfg.ldm_compression:
			raise NotImplementedError
		
		else:
			wav, org_sr = librosa.load(audio_fpath, sr=self.cfg.sample_rate)
			wav = torch.tensor(wav)
			lms = (self.to_melspecgram(wav) + torch.finfo().eps).log()
			lms = lms.unsqueeze(0)
			lms= trim_pad(self.cfg, lms)
			
			#try:
				#lms = torch.tensor(np.load(audio_fpath)).unsqueeze(0)
			#except ValueError:
				#pass
				#fname = np.random.choice(self.files_fsd50k)
				#audio_fpath = "data/FSD50K_lms/FSD50K.dev_audio/" + fname + ".npy"
				#lms = torch.tensor(np.load(audio_fpath)).unsqueeze(0)
			
			if self.norm_stats is not None:
				lms = (lms - self.norm_stats[0]) / self.norm_stats[1]
			# transforms
			if self.transform is not None:
				lms = self.transform(lms)
			return lms, label_indices

def calculate_norm_stats(dataset, n_norm_calc=10000):

	# calculate norm stats (randomly sample n_norm_calc points from dataset)
	idxs = np.random.randint(0, len(dataset), size=n_norm_calc)
	lms_vectors = []
	for i in tqdm(idxs):
		lms_vectors.append(dataset[i][0])
	lms_vectors = torch.stack(lms_vectors)
	norm_stats = float(lms_vectors.mean()), float(lms_vectors.std() + torch.finfo().eps)

	print(f'Dataset contains {len(dataset)} files with normalizing stats\n'
			f'mean: {norm_stats[0]}\t std: {norm_stats[1]}')
	norm_stats_dict = {'mean': norm_stats[0], 'std': norm_stats[1]}
	with open('norm_stats.json', mode='w') as jsonfile:
		json.dump(norm_stats_dict, jsonfile, indent=2)

from pydub import AudioSegment
import os

def extract_compressed_wav(audio_fpath, tmp_path, bitrate='32k', sr=16000):
	mp3_path = compress_to_mp3(audio_fpath, tmp_path, bitrate)
	wav_path = convert_to_wav(mp3_path, tmp_path)
	delete_file(mp3_path)
	wav, org_sr = librosa.load(wav_path, sr=sr)
	wav = torch.tensor(wav)
	delete_file(wav_path)
	return wav, org_sr

# Compress to MP3
def compress_to_mp3(input_path, output_directory, bitrate="192k"):
    audio = AudioSegment.from_wav(input_path)
    output_path_mp3 = os.path.join(output_directory, os.path.basename(input_path).replace(".wav", ".mp3"))
    #print(output_path_mp3)
    audio.export(output_path_mp3, format="mp3", bitrate=bitrate)
    return output_path_mp3

# Convert back to WAV
def convert_to_wav(mp3_path, output_directory):
    audio = AudioSegment.from_mp3(mp3_path)
    output_path_wav = os.path.join(output_directory, os.path.basename(mp3_path).replace(".mp3", ".wav"))
    audio.export(output_path_wav, format="wav")
    return output_path_wav

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        #print(f"{file_path} deleted successfully")
    #else:
        #print(f"{file_path} not found")

# Trim or pad
def trim_pad(cfg, lms):
	l = lms.shape[-1]
	if l > cfg.crop_frames:
		start = np.random.randint(l - cfg.crop_frames)
		lms = lms[..., start:start + cfg.crop_frames]
	elif l < cfg.crop_frames:
		pad_param = []
		for i in range(len(lms.shape)):
			pad_param += [0, cfg.crop_frames - l] if i == 0 else [0, 0]
		lms = F.pad(lms, pad_param, mode='constant', value=0)
	lms = lms.to(torch.float)
	#print(f"the shape of spectrogram is:{lms.shape}")
	return lms
if __name__ == "__main__":
	pass 
