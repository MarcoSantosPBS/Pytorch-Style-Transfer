import torch
import Processor
import numpy as np
from matplotlib import image
from torch.optim import Adam

class Transfer:

    def __init__(self, content_image, style_image, model):

        self.content_image = content_image
        self.style_image = style_image
        self.content_feature = self.__get_features(content_image, model)['conv4_2']
        self.style_features = self.__get_features(style_image, model)
        
        self.model = model
        self.style_layers_weights = {
            'conv1_1': 1.,
            'conv2_1': 0.8,
            'conv3_1': 0.7,
            'conv4_1': 0.5,
            'conv5_1': 0.2
        }
        self.style_grams = {layer : self.__get_gram_matrix(self.style_features[layer]) for layer in self.style_features}

        self.content_weight = 1
        self.style_weight = 1e-1
        

    def __get_features(self, image, model, layers = None):
        
        # layers {index_in_vgg19_model : layer_name}
        if layers is None:
            layers = {
                '0':'conv1_1',
                '5':'conv2_1',
                '10':'conv3_1',
                '19':'conv4_1',
                '21':'conv4_2',
                '28':'conv5_1'
            }

        
        features = {}
        x = image
        # models._modules.items() returns a dict with the layer index as key and the layer as value
        for index, layer in model._modules.items():
            x = layer(x)
            if index in layers:
                features[layers[index]] = x

        return features

    def __get_gram_matrix(self, tensor):

        _, depth, height, width = tensor.size()

        tensor = tensor.view(depth, height * width)
        
        return torch.mm(tensor, tensor.t())


    def __save_image(self, tensor, file_name):

        img = tensor.to('cpu').clone().detach()
        img = img.numpy().squeeze()
        img = img.transpose(1,2,0)
        img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        img = img.clip(0,1)
        
        image.imsave(file_name, img)


    def Run(self, epochs, save_after):

        target_image = self.content_image.clone().requires_grad_(True)

        optimizer = Adam([target_image], lr= 0.003)

        for i in range(epochs):
            target_features = self.__get_features(target_image, self.model)
            content_loss = torch.mean((target_features['conv4_2'] - self.content_feature)**2)

            style_loss = 0

            for layer in self.style_layers_weights:

                actual_target_layer_style = target_features[layer]
                target_gram = self.__get_gram_matrix(actual_target_layer_style)

                _, d, h, w = actual_target_layer_style.shape

                layer_weight = self.style_layers_weights[layer]
                
                layer_style_loss = layer_weight * torch.mean((target_gram - self.style_grams[layer])**2 )
                style_loss += layer_style_loss/ (d + h + w)
            
            total_loss = (self.content_weight * content_loss) + (self.style_weight * style_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % save_after == 0:
                self.__save_image(target_image, "target_image")
        
        return target_image