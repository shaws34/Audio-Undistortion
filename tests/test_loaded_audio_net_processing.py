import unittest
import os
import torch
import torchaudio

from training.audio_dataset import AudioDataset
from utils.audio_distortion import distort_audio
from training.audio_net import AudioNet


TEST_FILE = "C:/Users/samue/Documents/Sam's School/GitHub/Audio-Undistortion/training/prepared_data/noisy/distorted_0_0_1462-170142-0000.flac_16.0khz_24.0kframes.wav"
AUDIO_NET_FILE = "C:/Users/samue/Documents/Sam's School/GitHub/Audio-Undistortion/training/AudioNet.pth"
OUTPUT_UNDISTORTED_TEST_FILE = "C:/Users/samue/Documents/Sam's School/GitHub/Audio-Undistortion/tests/undistorted_test_file.wav"

class TestLoadedAudioNetProcessing(unittest.TestCase):
    def setUp(self) -> None:
        test_audio, sample_rate = torchaudio.load(TEST_FILE)
        test_audio = test_audio.to("cuda")
        print(test_audio.shape)


        self.audio_net = torch.load(AUDIO_NET_FILE).to("cuda")
        output = self.audio_net(test_audio)

        torchaudio.save(OUTPUT_UNDISTORTED_TEST_FILE, output.to("cpu"), self.audio_net.sample_rate)

        return super().setUp()
    
    def test_audio_net(self):
        self.assertTrue(os.path.exists(OUTPUT_UNDISTORTED_TEST_FILE))
        self.assertTrue(os.path.getsize(OUTPUT_UNDISTORTED_TEST_FILE) > 0)