import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Discriminator(nn.Module):
    def __init__(self, h_dim, output_dim):
        super(Discriminator, self).__init__()
        self.h_dim = h_dim

        self.feature_d = nn.Sequential()
        self.feature_d.add_module('df_conv1', nn.Conv2d(h_dim, h_dim*2, kernel_size=3, padding=1))
        self.feature_d.add_module('df_relu1', nn.ReLU(True))
        self.feature_d.add_module('df_conv2', nn.Conv2d(h_dim*2, h_dim*2, kernel_size=3, padding=1))
        self.feature_d.add_module('df_relu2', nn.ReLU(True))
        self.feature_d.add_module('df_conv3', nn.Conv2d(h_dim*2, h_dim*2, kernel_size=3, padding=1))
        self.feature_d.add_module('df_relu3', nn.ReLU(True))
        self.feature_d.add_module('df_conv4', nn.Conv2d(h_dim*2, h_dim*2, kernel_size=3, padding=1))
        self.feature_d.add_module('df_relu4', nn.ReLU(True))
        self.feature_d.add_module('df_conv5', nn.Conv2d(h_dim*2, h_dim*2, kernel_size=3, padding=1))
        self.feature_d.add_module('df_relu5', nn.ReLU(True))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(h_dim * 2 * 4 * 4, output_dim * 2))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(output_dim * 2, output_dim * 2))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(output_dim * 2, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, feature):
        feature_d = self.feature_d(feature).view(-1, self.h_dim * 2 * 4 * 4)
        domain_output = self.domain_classifier(feature_d)
        return domain_output


class ConvBlock(nn.Module):
    def __init__(self, h_dim):
        super(ConvBlock, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f1_conv', nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1))
        self.feature.add_module('f1_bn', nn.BatchNorm2d(h_dim))
        self.feature.add_module('f1_relu', nn.ReLU(True))

    def forward(self, feature):
        return self.feature(feature)


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.h_dim_1 = 64
        self.h_dim = 128
        self.output_dim = 256

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, self.h_dim_1, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(self.h_dim_1))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))

        self.feature.add_module('f_conv2', nn.Conv2d(self.h_dim_1, self.h_dim, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(self.h_dim))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.h_dim * 4 * 4, self.output_dim))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.output_dim))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(self.output_dim, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.feature_1 = ConvBlock(self.h_dim)
        self.feature_2 = ConvBlock(self.h_dim)
        self.feature_3 = ConvBlock(self.h_dim)
        self.feature_4 = ConvBlock(self.h_dim)
        self.feature_5 = ConvBlock(self.h_dim)

        self.d1 = Discriminator(self.h_dim, self.output_dim)
        self.d2 = Discriminator(self.h_dim, self.output_dim)
        self.d3 = Discriminator(self.h_dim, self.output_dim)
        self.d4 = Discriminator(self.h_dim, self.output_dim)
        self.d5 = Discriminator(self.h_dim, self.output_dim)


    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)

        feature = self.feature_1(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        d1_output = self.d1(reverse_feature)

        feature = self.feature_2(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        d2_output = self.d2(reverse_feature)

        feature = self.feature_3(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        d3_output = self.d3(reverse_feature)

        feature = self.feature_4(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        d4_output = self.d4(reverse_feature)

        feature = self.feature_5(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        d5_output = self.d5(reverse_feature)

        feature = feature.view(-1, self.h_dim * 4 * 4)
        class_output = self.class_classifier(feature)

        return class_output, [d1_output, d2_output, d3_output, d4_output, d5_output]
