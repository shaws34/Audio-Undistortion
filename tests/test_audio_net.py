import unittest
import torch

from training.audio_net import AudioNet

class TestAudioNet(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_audio_net_8000_frames_32_channels(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"using {device} device.")
        torch.cuda.empty_cache()
        target_output_size = 8000
        net = AudioNet(kernal_size=3, out_channels=32, target_output_size=target_output_size).to(device)
        x = torch.rand(1, target_output_size*3).to(device)
        print(x.shape)
        output = net(x)
        self.assertEqual(output.shape, torch.Size([1, target_output_size]))
        self.assertEqual(output.dtype, torch.float32)
        self.assertIn(str(device),str(output.device))

    def test_audio_net_8000_sample_rate_4000_frames_16_channels_5_kernal_size(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"using {device} device.")
        torch.cuda.empty_cache()
        sample_rate = 8000
        target_output_size = 4000
        kernal_size = 5
        out_channels = 16
        net = AudioNet(kernal_size=kernal_size, out_channels=out_channels, target_output_size=target_output_size, sample_rate=sample_rate).to(device)
        x = torch.rand(1, target_output_size*3).to(device)
        output = net(x)
        self.assertEqual(output.shape, torch.Size([1, target_output_size]))
        self.assertEqual(output.dtype, torch.float32)
        self.assertIn(str(device),str(output.device))
