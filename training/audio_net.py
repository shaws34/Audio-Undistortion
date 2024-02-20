import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioNet(nn.Module):
    
    def __init__(self, target_output_size=8000, sample_rate=16000, context=1, num_additional_convolutions=3, out_channels=32, kernal_size=3):
        super(AudioNet, self).__init__()
        self.output_size = target_output_size
        self.input_size = target_output_size + (2 * context * target_output_size)

        self.sample_rate = sample_rate
        self.context = context
        self.num_additional_convolutions = num_additional_convolutions
        self.out_channels = out_channels
        self.kernal_size = kernal_size
        self.stride = 1
        self.padding = (kernal_size - 1) // 2

        sr = self.sample_rate
        b_kernel_size = int(10 * sample_rate//sr) + 1
        b_padding = (b_kernel_size - 1) // 2

        self.base_layers = nn.ModuleList()
        self.base_layers.append(nn.Conv1d(1, self.out_channels, kernel_size=b_kernel_size, stride=self.stride, padding=b_padding))
        while sr > 400:
            sr = sr // 2
            b_kernel_size = int(10 * sample_rate//sr) + 1
            b_padding = (b_kernel_size - 1) // 2
            self.base_layers.append(nn.Conv1d(1, self.out_channels, kernel_size=b_kernel_size, stride=self.stride, padding=b_padding))
        
        length_of_base_layers = len(self.base_layers)
        self.pool1 = nn.MaxPool1d(kernel_size=self.kernal_size, stride=length_of_base_layers, padding=self.padding)
        

        self.deeper_layers = nn.Sequential()
        for _ in range(self.num_additional_convolutions):
            self.deeper_layers.append(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=self.kernal_size, stride=self.stride, padding=self.padding))
            self.deeper_layers.append(nn.ReLU())

        self.batch_norm = nn.BatchNorm1d(self.out_channels)
        self.pool2 = nn.MaxPool1d(kernel_size=self.kernal_size, stride=num_additional_convolutions, padding=self.padding)
        self.fc1 = nn.Linear(self.out_channels, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(0)

        x = [layer(x) for layer in self.base_layers]
        x = torch.cat(x, 0)
        x = self.pool1(x.t())
        x = x.t()
        x = self.deeper_layers(x)
        x = self.batch_norm(x.t())
        x = self.pool2(x.t())
        x = self.fc1(x.t())
        return x.t()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    torch.cuda.empty_cache()
    net = AudioNet(kernal_size=3, out_channels=32).to(device)
    print(net)
    for _ in range(10):
        x = torch.rand(1, 1, 24000).to(device)
        print(x.shape)
        output = net(x)
        print(output)


