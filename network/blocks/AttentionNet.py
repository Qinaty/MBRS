import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)

    def forward(self, x):
        x1 = self.conv2(torch.relu(self.conv1(x)))
        out = x+x1
        return out


class Non_local_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Non_local_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.out_channel, -1)  # (b, c, h*w)
        g_x = g_x.permute(0, 2, 1)  # (b, h*w, c)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1)  # (b, c, h*w)
        theta_x = theta_x.permute(0, 2, 1)  # (b, h*w, c)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)  # (b, c, h*w)

        f1 = torch.matmul(theta_x, phi_x)  # (b, h*w, h*w)
        f_div_C = torch.softmax(f1, dim=-1)
        y = torch.matmul(f_div_C, g_x)  # (b, h*w, c)
        # 调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
        y = y.permute(0, 2, 1).contiguous()  # (b, c, h*w)
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z