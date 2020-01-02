from torch import nn


class AlexNet(nn.Module):
    def __init__(self, ):
        super(AlexNet, self).__init__()
        self.sharedFeatExtra = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            # conv3
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )
        self.channel_adjust = nn.Conv2d(256, 512, 1)

    def forward(self, input):
        output = self.sharedFeatExtra(input)
        output = self.channel_adjust(output)
        return output
