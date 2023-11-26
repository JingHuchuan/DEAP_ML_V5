import torch.nn as nn
import torch
import torch.nn.functional as F


class EEGNet(nn.Module):

    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        # self.drop_out = 0.5
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            # nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
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
                kernel_size=(1, 1),  # filter size
                # kernel_size=(14, 1),  # filter size
                # kernel_size=(8, 1),  # filter size
                # kernel_size=(2, 1),  # filter size
                groups=8,
                bias=True

            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 8),  # filter size
                groups=16,
                bias=True
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=True
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear(48, classes_num)

    # # 定义权值初始化
    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             torch.nn.init.xavier_normal_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             torch.nn.init.normal_(m.weight.data, 0, 0.1)
    #             # m.weight.record.normal_(0,0.1)
    #             m.bias.data.zero_()

    def forward(self, x):
        # print('123123',x.shape)
        x = self.block_1(x)

        # print("block1", x.shape)
        x = self.block_2(x)

        # print("block2", x.shape)
        x = self.block_3(x)
        x_features = x.view(x.size(0), -1)
        # print("block3", x.shape)

        x = x.view(x.size(0), -1)
        # x_feature = x
        x = self.out(x)

        return F.softmax(x, dim=1), x_features  # return x for visualization
