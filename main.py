import torch
from torch.optim import Adam
from torchvision import models
from Processor import ImageProcessor
from StyleTransfer import Transfer

use_cuda = torch.cuda.is_available()

vgg = models.vgg19(pretrained= True).features

for param in vgg.parameters():
    param.requires_grad_(False)

if use_cuda: vgg = vgg.to('cuda')

processor =  ImageProcessor('Imagens/Cachorro.jpg','Estilos/Tsunami_by_hokusai_19th_century.jpg')

content_tensor, style_tensor = processor.load_images()

tf = Transfer(content_tensor, style_tensor, vgg)

tf.Run(5000,500)