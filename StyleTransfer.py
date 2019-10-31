import torch
from torch.optim import Adam

class Transfer:

    def __init__(self, content_image, style_image, model):

        self.content_feature = self.get_features(content_image, model)['conv4_2']
        self.style_features = self.get_features(style_image, model)
        
        self.model = model
        self.style_layers_weights = {
            'conv1_1': 1.,
            'conv2_1': 0.8,
            'conv3_1': 0.7,
            'conv4_1': 0.5,
            'conv5_1': 0.2
        }
        self.style_grams = {layer : get_gram_matrix(self.style_features[layer]) for layer in self.style_features}

        self.content_weight = 1
        self.style_weight = 1e-1
        

    def __get_features(self, image, model, layers = None):
        
        # layers {index_in_vgg19_model : layer_name}
        if layers:
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
        for index, layer in models._module.items():
            x = layer(x)
            if index in layers:
                features[layers[index]] = x

        return features

    def __get_gram_matrix(self, tensor):

        _, depth, height, width = tensor.size()

        tensor = tensor.view(depth, height * width)
        
        return torch.mm(tensor, tensor.t())

    def Run(self, epochs):

        target_image = content_image.clone().requires_grad_(True)

        optimizer = Adam([target], lr= 0.003)

        for i in range(epochs):
            
            target_features = self.get_features(target_image, self.model)
            content_loss = torch.mean((target_feature['conv4_2'] - self.content_feature)**2)

            style_loss = 0

            for layer in self.style_layers_weights:

                actual_target_layer_style = target_features[layer]
                target_gram = self.get_gram_matrix(actual_target_layer_style)

                _, d, h, w = actual_target_layer_style.shape

                layer_weight = self.style_layers_weights[layer]
                
                layer_style_loss += layer_weight * torch.mean((target_gram - self.style_grams[layer])**2 )
                style_loss += layer_style_loss/ (d + h + w)
            
            total_loss = (self.content_weight * content_loss) + (self.style_weight * style_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return target_image