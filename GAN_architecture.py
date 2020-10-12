import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for our testing purposes, please do not change!




class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (CelebA is rgb, so 3 is the default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, stride=1, padding=0),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            padding: the amount of implicit zero-paddings on both sides of the output feature representation
            bias: a boolean, false meaning no bias included
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)





# class Discriminator(nn.Module):
#     '''
#     Discriminator Class
#     Values:
#         im_chan: the number of channels in the images, fitted for the dataset used, a scalar
#               (CelebA is rgb, so 3 is our default)
#         hidden_dim: the inner dimension, a scalar
#     '''
#     def __init__(self, im_chan=3, hidden_dim=64):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             self.make_disc_block(im_chan, hidden_dim),
#             self.make_disc_block(hidden_dim, hidden_dim * 2),
#             self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
#             self.make_disc_block(hidden_dim * 4, hidden_dim * 8),
#             self.make_disc_block(hidden_dim * 8, 1, stride=1, padding=0, final_layer=True),
#         )

#     def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
#         '''
#         Function to return a sequence of operations corresponding to a discriminator block of DCGAN;
#         a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
#         Parameters:
#             input_channels: how many channels the input feature representation has
#             output_channels: how many channels the output feature representation should have
#             kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
#             stride: the stride of the convolution
#             final_layer: a boolean, true if it is the final layer and false otherwise 
#                       (affects activation and batchnorm)
#         '''
#         if not final_layer:
#             return nn.Sequential(
#                 nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
#                 nn.BatchNorm2d(output_channels),
#                 nn.LeakyReLU(0.2, inplace=True),
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
#             )

#     def forward(self, image):
#         '''
#         Function for completing a forward pass of the discriminator: Given an image tensor, 
#         returns a 1-dimension tensor representing fake/real.
#         Parameters:
#             image: a flattened image tensor with dimension (im_chan)
#         '''
#         disc_pred = self.disc(image)
#         return disc_pred.view(len(disc_pred), -1)

class Classifier(nn.Module):
    '''
    Classifier Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (CelebA is rgb, so 3 is your default)
        n_classes: the total number of classes in the dataset, an integer scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=3, n_classes=2, hidden_dim=64):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            self.make_classifier_block(im_chan, hidden_dim),
            self.make_classifier_block(hidden_dim, hidden_dim * 2),
            self.make_classifier_block(hidden_dim * 2, hidden_dim * 4, stride=3),
            self.make_classifier_block(hidden_dim * 4, n_classes, final_layer=True),
        )

    def make_classifier_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a classifier block; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the classifier: Given an image tensor, 
        returns an n_classes-dimension tensor representing classes.
        Parameters:
            image: a flattened image tensor with im_chan channels
        '''
        class_pred = self.classifier(image)
        return class_pred.view(len(class_pred), -1)