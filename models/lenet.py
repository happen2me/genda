# modified from 'https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py'
import torch.nn as nn
from collections import OrderedDict

DEBUG = False


def print_detail(layer, img):
    print("in {}th layer".format(layer))
    print("img shape is:")
    print(img.shape)


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        # Layer 1: Convolutional. Input_channel = 1. Output_channel = 6.
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        if DEBUG:
            print_detail(0, img)
        output = self.c1(img)
        if DEBUG:
            print_detail(1, output)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()
        # Layer 2: Convolutional. Output = 10x10x16.
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        if DEBUG:
            print_detail(2, output)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        if DEBUG:
            print_detail(3, output)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        if DEBUG:
            print_detail(4, output)
        return output


class F5_linear(nn.Module):
    def __init__(self):
        super(F5_linear, self).__init__()
        # Layer 5: Fully Connected. Input = 84. Output = 10.
        self.f5_linear = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10))
        ]))

    def forward(self, img):
        if DEBUG:
            print("in 5th(linear) layer")
        output = self.f5_linear(img)
        return output


class F5_softmax(nn.Module):
    def __init__(self):
        super(F5_softmax, self).__init__()
        # Layer 5: Fully Connected. Input = 84. Output = 10.
        self.f5_softmax = nn.Sequential(OrderedDict([
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        if DEBUG:
            print("in 5th(softmax) layer")
        output = self.f5_softmax(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5_linear = F5_linear()
        self.f5_softmax = F5_softmax()

    # def forward(self, img):
    #     output = self.c1(img)
    #
    #     x = self.c2_1(output)
    #     output = self.c2_2(output)
    #
    #     output += x
    #
    #     output = self.c3(output)
    #     output = output.view(img.size(0), -1)
    #     feature = output.view(-1, 120)
    #     output = self.f4(output)
    #     output = self.f5_linear(output)
    #     output = self.f5_softmax(output)
    #     return output

    def forward(self, img, out_feature=False):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        feature = output.view(-1, 120)
        output = self.f4(output)
        output = self.f5_linear(output)
        output = self.f5_softmax(output)
        if out_feature:
            return output, feature
        return output


class LeNet5ClassifierActivation(nn.Module):
    """
    Input: from layer f4
    output: the output of f5 linear
    """
    def __init__(self):
        super(LeNet5ClassifierActivation, self).__init__()
        self.f5_linear = F5_linear()

    def forward(self, features):
        output = self.f5_linear(features)
        return output


class LeNet5Classifier(nn.Module):
    """
    Input: from layer f4
    output: 10
    """
    def __init__(self):
        super(LeNet5Classifier, self).__init__()
        self.f5_linear = F5_linear()
        self.f5_softmax = F5_softmax()

    def forward(self, features):
        output = self.f5_linear(features)
        output = self.f5_softmax(output)
        return output


class LeNet5Encoder(nn.Module):
    """
        Input - 1x32x32
        Output - 84
        """

    def __init__(self):
        super(LeNet5Encoder, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        return output

