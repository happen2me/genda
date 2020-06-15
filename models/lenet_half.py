import torch.nn as nn
from collections import OrderedDict

dropout_rate = 0.0


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        # Layer 1: Convolutional. Input_channel = 1. Output_channel = 3.
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 3, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('d1', nn.Dropout2d(p=dropout_rate)),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()
        # Layer 2: Convolutional. Output = 10x10x8.
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(3, 8, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('d2', nn.Dropout2d(p=dropout_rate)),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        # Layer 3: Fully Connected. Input = 200. Output = 120.
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(8, 60, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('d3', nn.Dropout2d(p=dropout_rate))
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(60, 42)),
            ('relu4', nn.ReLU()),
            ('d4', nn.Dropout(p=dropout_rate))
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()
        # Layer 5: Fully Connected. Input = 84. Output = 10.
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(42, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5Half(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.c1 = C1()
        self.c2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img, out_feature=False):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)

        output = output.view(img.size(0), -1)
        output = self.f4(output)
        feature = output.view(-1, 42)
        output = self.f5(output)
        if out_feature:
            return output, feature
        return output


class LeNet5HalfEncoder(nn.Module):
    """
    Input - 1x32x32
    Output - 84
    """
    def __init__(self):
        super(LeNet5HalfEncoder, self).__init__()

        self.c1 = C1()
        self.c2 = C2()
        self.c3 = C3()
        self.f4 = F4()

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        return output


class LeNet5HalfClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.f5 = F5()

    def forward(self, img):
        output = self.f5(img)
        return output
