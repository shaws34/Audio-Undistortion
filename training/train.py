import os
import torch
import torch.nn as nn
import torch.optim as optim
from training.audio_dataset import AudioDataset
from training.audio_net import AudioNet
from utils.audio_distortion import distort_audio
import matplotlib.pyplot as plt


TARGET_OUTPUT_SIZE = 8000
SAMPLE_RATE = 16000
CONTEXT = 1
NUM_ADDITIONAL_CONVOLUTIONS = 3
OUT_CHANNELS = 32
KERNAL_SIZE = 3
NEURAL_NETWORK_FILE = "training/saved_models/AudioNet_MSELoss_1000_slower_rate_higher_decay_higher_eps.pth"
LOSS_FUNCTION = nn.MSELoss()
DATA_SET_FILE = "training/AudioSet_14259.pth"
DATA_DIRECTORY = "training/original_data/LibriSpeech/dev-clean"
OUTPUT_DIRECTORY = "training/prepared_data"
ACCEPT_EXTENSIONS=[".flac",".wav"]
DISTORT_FUNCTION = distort_audio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20

def train():
    if os.path.exists(NEURAL_NETWORK_FILE):
        net = torch.load(NEURAL_NETWORK_FILE)
    else:
        net = AudioNet(target_output_size=TARGET_OUTPUT_SIZE, sample_rate=SAMPLE_RATE, context=CONTEXT, num_additional_convolutions=NUM_ADDITIONAL_CONVOLUTIONS, out_channels=OUT_CHANNELS, kernal_size=KERNAL_SIZE)
        torch.save(net, NEURAL_NETWORK_FILE)
    net.to(DEVICE)
    print(net)
    print(f"Loaded NN. Using {DEVICE} for training")
    print(f"Beginning data set creation. This may take a while...")
    if os.path.exists(DATA_SET_FILE):
        trainset = torch.load(DATA_SET_FILE)
    else:
        trainset = AudioDataset(data_directory=DATA_DIRECTORY, output_directory=OUTPUT_DIRECTORY, distort_function=DISTORT_FUNCTION, input_frame_size=TARGET_OUTPUT_SIZE, input_frame_context_size=CONTEXT, data_sample_rate=SAMPLE_RATE, max_data_files=None, accept_extensions=ACCEPT_EXTENSIONS)
        torch.save(trainset, f'training/AudioSet_{len(trainset)}.pth')
    print(f"Data set created. Using {len(trainset)} samples for training")

    # train_data = torch.utils.data.DataLoader(trainset, shuffle=True)
    # test_data = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

    criterion = LOSS_FUNCTION
    optimizer = optim.Adam(net.parameters(), eps=.1, lr=0.0001, weight_decay=0.01)
    optimizer.zero_grad()

    log_interval = 1000

    losses = []
    # Training loop
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (clean_data, noisy_data) in enumerate(trainset,1):
            clean_data = clean_data.to(DEVICE)
            noisy_data = noisy_data.to(DEVICE)
        
            # forward + backward + optimize
            outputs = net(noisy_data)
            ideal_output = clean_data[:,CONTEXT*TARGET_OUTPUT_SIZE:-(CONTEXT*TARGET_OUTPUT_SIZE)]
            loss = criterion(outputs, ideal_output)
            # print(f"Loss: {loss} - Running Loss: {running_loss}")

            loss.backward()
            
            # print statistics
            running_loss += loss.item()

            if i % log_interval == 1 or i == len(trainset):    # print every log_interval
                losses.append(running_loss / log_interval)

                optimizer.step()
                optimizer.zero_grad()

                print('[%d, %5d] loss: %.7f' %
                    (epoch + 1, i, running_loss / log_interval))
                running_loss = 0.0
                torch.save(net, NEURAL_NETWORK_FILE)
    
    print('Finished Training')
    plt.plot(losses)
    plt.show()
