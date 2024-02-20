import torch
from audio_distortion import distort_audio

audio = torch.randn(1, 8000)
print(audio.shape)
distorted_audio = distort_audio(audio)
