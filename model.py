import torch.nn as nn

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # [16, 150, 150]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [16, 75, 75]

            nn.Conv2d(16, 32, 3, 1, 1),  # [32, 75, 75]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [32, 36, 36]

            nn.Conv2d(32, 64, 3, 1, 1),  # [64, 36, 36]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 18, 18]

            nn.Conv2d(64, 128, 3, 1),  # [128, 18, 18]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 9, 9]

            nn.Conv2d(128, 128, 3, 1, 1),  # [128, 9, 9]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        out = self.cnn(x)
        
        out = out.view(out.size()[0], -1)
        #print(out.size()[1])
        return self.fc(out)