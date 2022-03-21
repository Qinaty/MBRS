from . import *


class Generate_Mask(nn.Module):
    def __init__(self, H, W, blocks=4, channels=64):
        super(Generate_Mask, self).__init__()
        self.H = H
        self.W = W

        self.conv1 = nn.Conv2d(3, channels, 3, 1, 1)  # add channels
        self.conv2 = nn.Conv2d(3 + channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(3 + 2 * channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.senet = SENet(3 + 2 * channels, 3 + 2 * channels, blocks=blocks)

    def forward(self, image):
        x1 = self.lrelu(self.conv1(image))
        x2 = self.lrelu(self.conv2(torch.cat((image, x1), 1)))
        x = torch.cat((image, x1, x2), 1)
        x = self.senet(x)
        mask = self.conv3(x)

        return mask