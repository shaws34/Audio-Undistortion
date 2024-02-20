import unittest
import os
from training.audio_dataset import AudioDataset
from utils.audio_distortion import distort_audio


class TestAudioDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.data_directory = "training/original_data/LibriSpeech/dev-clean"
        self.output_directory = "tests/prepared_data"
        if os.path.exists(self.output_directory):
            for root, _, files in os.walk(self.output_directory):
                for file in files:
                    os.remove(os.path.join(root, file))
            
        self.input_frame_size = 8000
        self.input_frame_context_size = 1
        self.data_sample_rate = 16000
        self.max_data_files = 10
        self.accept_extensions = [".flac", ".wav"]

        self.trainset = AudioDataset(self.data_directory, self.output_directory, distort_audio, self.input_frame_size, self.input_frame_context_size, self.data_sample_rate, self.max_data_files, self.accept_extensions)
        
    def tearDown(self) -> None:
        for file in os.listdir("tests/prepared_data/clean"):
            os.remove(f"tests/prepared_data/clean/{file}")
        for file in os.listdir("tests/prepared_data/noisy"):    
            os.remove(f"tests/prepared_data/noisy/{file}")
        os.rmdir("tests/prepared_data/clean")
        os.rmdir("tests/prepared_data/noisy")
        os.rmdir("tests/prepared_data")
    
    def test_audio_dataset_len(self):
        self.assertEqual(len(self.trainset), 10)
    
    def test_audio_dataset_getitem(self):
        clean_file, noisy_file = self.trainset[0]
        self.assertTrue(clean_file.shape == (self.trainset.data_channels, self.trainset.get_data_length()))
        self.assertTrue(noisy_file.shape == (self.trainset.data_channels, self.trainset.get_data_length()))

    def test_audio_dataset_load_from_directory(self):
        self.trainset = AudioDataset(data_directory=self.output_directory, load_from_directory=True)
        self.assertEqual(len(self.trainset), 10)

    def test_audio_dataset_iter(self):
        for i, (clean_file, noisy_file) in enumerate(self.trainset):
            self.assertTrue(clean_file.shape == (self.trainset.data_channels, self.trainset.get_data_length()))
            self.assertTrue(noisy_file.shape == (self.trainset.data_channels, self.trainset.get_data_length()))
            if i > 10:
                break


