import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class SelfNet(nn.Module):

    def __init__(self, routing_iterations, classes_num):
        super(SelfNet, self).__init__()
        # self.drop_out = 0.5
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            # nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=4,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=True
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(32, 1),  # filter size
                # kernel_size=(8, 1),  # filter size
                # kernel_size=(15, 1),  # filter size
                # kernel_size=(4, 1),  # filter size
                # kernel_size=(1, 1),  # filter size
                # kernel_size=(5, 1),  # filter size
                # kernel_size=(1, 1),  # filter size
                groups=8,
                bias=True

            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.primaryCaps = PrimaryCapsLayer(16, 32, 8, kernel_size=(1, 8), stride=1)
        self.num_primaryCaps = 288
        routing_module = AgreementRouting(self.num_primaryCaps, classes_num, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, classes_num, 16, routing_module)
        self.out = nn.Linear(32, classes_num)

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x_middle = x
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        x_features = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1), x_features, x_middle


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        # 激活函数
        out = squash(out)
        return out


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x
