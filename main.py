import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import models
from Processor import ImageProcessor
from StyleTransfer import Transfer

use_cuda = torch.cuda.is_available()

vgg = models.vgg19(pretrained= True).features

for param in vgg.parameters():
    param.requires_grad_(False)

if use_cuda: vgg = vgg.to('cuda')

ImageProcessor('','')

content_tensor, style_tensor = ImageProcessor.load_images()

tf = Transfer(content_tensor, style_tensor, vgg)

tf.Run(5000)