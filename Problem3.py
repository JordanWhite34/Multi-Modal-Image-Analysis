# Problem 3: GAN
# In this problem, we will train a generative adversarial network (GAN) to generate new celebrities after showing it pictures of many real celebrities. 
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# %matplotlib inline

from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000)  # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
### (a) Prepare CelebA Dataset

# we will use the [Celeb-A Faces
# dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which can
# be downloaded from the [link](https://drive.google.com/file/d/1JgNzlZKQVlVwmlVx-5y6m4Dftbia51YB/view?usp=sharing).
# The dataset will download as a file named `img_align_celeba.zip`. Once
# downloaded, create a directory named `CelebA` and extract the zip file
# into that directory. Then, set the `dataroot` input for this notebook to
# the `CelebA` directory you just created. The resulting directory
# structure should be:

# ```
#    /path/to/CelebA
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# ```

# This is an important step because we will be using the ImageFolder
# dataset class, which requires there to be subdirectories in the
# dataset’s root folder. 
# !gdown --fuzzy https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ img_align_celeba.zip
# !mkdir ./CelebA
# !unzip ./content/img_align_celeba.zip -d ./CelebA
# Set parameters for the implemented network.
# Root directory for dataset
dataroot = "./CelebA"
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 256
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Learning rate for optimizers
lr = 0.0005
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Now, we can create the dataset, create the
# dataloader, set the device to run on, and finally visualize some of the
# training data.


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[192:], padding=2, normalize=True).cpu(),(1,2,0)))
### (b) GAN Implementation

#### 1. Weight Initialization 

# All model weights in GAN should be randomly initialized from a Normal distribution with `mean = 0`, `stdev = 0.02`. The ``weights_init`` function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to meet this criteria. This function is applied to the models immediately after initialization.
# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)
#### 2. Generator

# The generator $G$, is designed to map the latent space vector
# $z$ to data-space. Since our data are images, converting
# $z$ to data-space means ultimately creating an RGB image with the
# same size as the training images (i.e. 3x64x64). In practice, this is
# accomplished through a series of strided two dimensional convolutional
# transpose layers, each paired with a 2D batch norm layer and a relu
# activation. The output of the generator is fed through a tanh function
# to return it to the input data range of $[-1,1]$. 
class Generator(nn.Module):
  def __init__(self, latent_dim=100):
    super(Generator, self).__init__()
    # TODO: finish implementing the network architecture
    self.seq = nn.Sequential(
      # TODO: the first ConvTranspose2d layer will take the celebA images as input with size of 3x64x64
      # so you will need at least one ConvTranspose2d layer, one BatchNorm2d layer followed by a ReLU activation here
      nn.ConvTranspose2d(latent_dim, 512, kernel_size=(4, 4), stride=(1,1), padding=(0,0), bias=False),
      nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),

      # ENDS HERE 
      nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
      nn.Tanh()
    )

  def forward(self, input):
    x = self.seq(input)
    return x
# Create the generator
netG = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
#### 3. Discriminator

# The discriminator $D$ is a binary classification
# network that takes an image as input and outputs a scalar probability
# that the input image is real (as opposed to fake). Here, $D$ takes
# a 3x64x64 input image, processes it through a series of Conv2d,
# BatchNorm2d, and ReLU layers, and outputs the final probability
# through a Sigmoid activation function. This architecture can be extended
# with more layers if necessary for the problem, but there is significance
# to the use of the strided convolution, BatchNorm, and ReLUs. 

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    # TODO: finish implementing the network architecture
    self.seq = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        # TODO: the last Conv2d layer of the doscriminator followed by a sigmoid 
        nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1,1), padding=(0,0), bias=False),
        nn.Sigmoid()
        # ENDS HERE
    )

  def forward(self, input):
    x = self.seq(input)
    return x
# Create the Discriminator
netD = Discriminator().to(device)
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
#### Loss Functions and Optimizers

# With $D$ and $G$ setup, we can specify how they learn
# through the loss functions and optimizers. We will use the Binary Cross
# Entropy loss
# ([`BCELoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss))
# function which is defined in PyTorch as:

# \begin{align}\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]\end{align}

# Notice how this function provides the calculation of both log components
# in the objective function (i.e. $\log(D(x))$ and
# $\log(1-D(G(z)))$). We can specify what part of the BCE equation to
# use with the $y$ input. This is accomplished in the training loop
# which is coming up soon, but it is important to understand how we can
# choose which component we wish to calculate just by changing $y$
# (i.e. GT labels).
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
#### 5. Training Loop


# Finally, now that we have all of the parts of the GAN framework defined,
# we can train it. Be mindful that training GANs is somewhat of an art
# form, as incorrect hyperparameter settings lead to mode collapse with
# little explanation of what went wrong. Here, we will closely follow
# Algorithm 1 from Goodfellow’s paper, while abiding by some of the best
# practices shown in [`ganhacks`](https://github.com/soumith/ganhacks).
# Namely, we will construct different mini-batches for real and fake
# images, and also adjust G’s objective function to maximize
# $\log (D(G(z)))$. Training is split up into two main parts. Part 1
# updates the Discriminator and Part 2 updates the Generator.

**Part 1: Train the Discriminator**

# Recall, the goal of training the discriminator is to maximize the
# probability of correctly classifying a given input as real or fake. In
# terms of Goodfellow, we wish to “update the discriminator by ascending
# its stochastic gradient”. Practically, we want to maximize
# $\log(D(x)) + \log(1-D(G(z)))$. Due to the separate mini-batch
# suggestion from ganhacks, we will calculate this in two steps. First, we
# will construct a batch of real samples from the training set, forward
# pass through $D$, calculate the loss ($\log(D(x))$), then
# calculate the gradients in a backward pass. Secondly, we will construct
# a batch of fake samples with the current generator, forward pass this
# batch through $D$, calculate the loss ($\log(1-D(G(z)))$),
# and *accumulate* the gradients with a backward pass. Now, with the
# gradients accumulated from both the all-real and all-fake batches, we
# call a step of the Discriminator’s optimizer.

**Part 2: Train the Generator**

# As stated in the original paper, we want to train the Generator by
# minimizing $\log(1-D(G(z)))$ in an effort to generate better fakes.
# As mentioned, this was shown by Goodfellow to not provide sufficient
# gradients, especially early in the learning process. As a fix, we
# instead wish to maximize $\log(D(G(z)))$. In the code we accomplish
# this by: classifying the Generator output from Part 1 with the
# Discriminator, computing G’s loss *using real labels as GT*, computing
# G’s gradients in a backward pass, and finally updating G’s parameters
# with an optimizer step. It may seem counter-intuitive to use the real
# labels as GT labels for the loss function, but this allows us to use the
# $\log(x)$ part of the BCELoss (rather than the $\log(1-x)$
# part) which is exactly what we want.

# Finally, we will do some statistic reporting and at the end of each
# epoch we will push our fixed_noise batch through the generator to
# visually track the progress of G’s training. The training statistics
# reported are:

# -  **Loss_D** - discriminator loss calculated as the sum of losses for
#    the all real and all fake batches ($\log(D(x)) + \log(D(G(z)))$).
# -  **Loss_G** - generator loss calculated as $\log(D(G(z)))$
# -  **D(x)** - the average output (across the batch) of the discriminator
#    for the all real batch. This should start close to 1 then
#    theoretically converge to 0.5 when G gets better. Think about why
#    this is.
# -  **D(G(z))** - average discriminator outputs for the all fake batch.
#    The first number is before D is updated and the second number is
#    after D is updated. These numbers should start near 0 and converge to
#    0.5 as G gets better. Think about why this is.

# **Note:** This step might take a while, depending on how many epochs you
# run and if you removed some data from the dataset.


# Training Loop
num_epochs = 30

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ######################################################
        # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ######################################################
        
        # Initialize the gradient of D
        netD.zero_grad()

        # 1.1 Calculate the loss with all-real batch
        
        # Format batch
        real_batch = data[0].to(device)
        b_size = real_batch.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_batch).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # 1.2 Calculate the loss with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G and a label array
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        # Calculate the mean of output, D_G_z1, from netD with your fake batch
        D_G_z1 = output.mean().item()
        
        # 1.3 Combine the loss from all-real batch and all-fake batch
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D with step()
        optimizerD.step()

        ######################################################
        # 2. Update G network: maximize log(D(G(z)))
        ######################################################

        # TODO: Initialize the gradient of G
        netG.zero_grad()

        # TODO: create the label array, remember fake labels are real for generator cost
        label.fill_(real_label)

        # TODO: Since we just updated D, perform another forward pass of all-fake batch through D as output
        output = netD(fake).view(-1)

        # TODO: Calculate G's loss based on this output
        errG = criterion(output, label)

        # TODO: Calculate gradients using backward() for G's loss 
        errG.backward()
        
        # TODO: Calculate the mean of output, D_G_z2, from netD with your fake batch
        D_G_z2 = output.mean().item()

        # TODO: Update G with step
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
fig = plt.figure(figsize=(8,8))
plt.axis("off")
plt.rcParams['animation.embed_limit'] = 70.0  # Set the limit to 50 MB
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
