import os
from pathlib import Path
import torch
import torchaudio
from torch.utils import data

class AudioDataset(data.Dataset):
    def __init__(self, data_directory, output_directory=None, distort_function=None, input_frame_size=8000, input_frame_context_size=1, data_sample_rate=16000, max_data_files=None, accept_extensions=[".wav"], load_from_directory=None):
        assert os.path.exists(data_directory), f"Data directory {data_directory} does not exist"
        assert torchaudio.list_audio_backends() is not None, "No audio backend found, check torchaudio installation. Windows requires soundfile backend. Linux requires sox backend. Nvidia Cuda's ffmpeg library for GPU processing is also supported."

        super(AudioDataset, self).__init__()

        if load_from_directory is not None:
            self._load_from_directory(data_directory, accept_extensions)
            return

        Path(f"{output_directory}/clean").mkdir(parents=True, exist_ok=True)
        Path(f"{output_directory}/noisy").mkdir(parents=True, exist_ok=True)

        self.data_directory = data_directory 
        self.output_directory = output_directory
        self.distort_function = distort_function

        self.input_frame_size = input_frame_size
        self.input_frame_context_size = input_frame_context_size
        self.data_sample_rate = data_sample_rate
        self.accept_extensions = accept_extensions
        
        # Number of channels in the audio data, only 1 channel is supported
        self.data_channels=1

        self.data = []

        full_input_size = self.get_data_length()

        for root, dirs, files in os.walk(data_directory):
            for count, file in enumerate(files):
                if file.endswith(tuple(accept_extensions)):
                    data_file_path = os.path.join(root, file)
                    with open(data_file_path, "rb") as f:
                        waveform, sample_rate = torchaudio.load(f, format=data_file_path.split(".")[-1])
                        for i in range(0, waveform.shape[1], full_input_size):
                            if sample_rate != self.data_sample_rate:
                                waveform = torchaudio.transforms.Resample(sample_rate, self.data_sample_rate)(waveform)
                            audio = waveform[:, i:i+full_input_size]
                            audio = self.trim_audio(audio, full_input_size)
                            noisy_audio = self.distort_audio(audio)
                            
                            audio_file_name = f"{count}_{i}_{file}_{self.data_sample_rate / 1000}khz_{full_input_size / 1000}kframes.wav"
                            clean_audio_file_path = os.path.join(output_directory, "clean", audio_file_name)
                            noisy_audio_file_path = os.path.join(output_directory, "noisy", f"distorted_{audio_file_name}")

                            torchaudio.save(clean_audio_file_path, audio, self.data_sample_rate)
                            torchaudio.save(noisy_audio_file_path, noisy_audio, self.data_sample_rate)

                            self.data.append((clean_audio_file_path, noisy_audio_file_path))
                            if max_data_files is not None and len(self.data) >= max_data_files:
                                return
                            
    def _load_from_directory(self, load_from_directory, accept_extensions=[".wav"]):
        assert os.path.exists(load_from_directory), f"Data directory {load_from_directory} does not exist"
        assert os.path.exists(f"{load_from_directory}/clean"), f"Data directory {load_from_directory}/clean does not exist"
        assert os.path.exists(f"{load_from_directory}/noisy"), f"Data directory {load_from_directory}/noisy does not exist"
        self.data = []

        for root, dirs, files in os.walk(f"{load_from_directory}/clean"):
            for count, file in enumerate(files):
                if file.endswith(tuple(accept_extensions)):
                    clean_audio_file_path = os.path.join(root, file)
                    noisy_audio_file_path = os.path.join(root.replace("clean", "noisy"), file)
                    self.data.append((clean_audio_file_path, noisy_audio_file_path))

    def trim_audio(self, audio, target_length):
        if audio.shape[1] > target_length:
            audio = audio[:, :target_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[1]))
        return audio
    
    def distort_audio(self, audio):
        return self.distort_function(audio)
    
    def get_data_length(self):
        return self.input_frame_size * (2 * self.input_frame_context_size + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean_audio_file_path, noisy_audio_file_path = self.data[idx]
        clean_audio, _ = torchaudio.load(clean_audio_file_path)
        noisy_audio, _ = torchaudio.load(noisy_audio_file_path)
        return clean_audio, noisy_audio
    
    def __iter__(self):
        for i in range(len(self.data)):
            yield self.__getitem__(i)

    
    



