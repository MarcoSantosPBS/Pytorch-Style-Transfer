from PIL import Image
from torchvision import transforms

class ImageProcessor:

    def __init__(self, contentImagePath, styleImagePath):
        self.content_image_path = contentImagePath
        self.style_image_path = styleImagePath
        
    
    def load_images(self, max_dimensions = (400,400) , transform = None):
        ''' 
        Load Content Image, Style Image into an Torch Tensor

        Parameters
        ----------
        max_dimensions : int
            The max dimensions of the returned images

        transform : torchvision.transforms
            A custom transform if desired.

        
        Return : torch.Tensor, torch.Tensor
            CI and SI as torch tensors
        '''

        content_image = Image.open(self.content_image_path)
        style_image = Image.open(self.style_image_path)

        CI_size = content_image.size
        SI_size = style_image.size

        if (CI_size or SI_size) > max_dimensions:
            new_size = max_dimensions
        
        else:
            new_size = CI_size if max(CI_size) > max(SI_size) else SI_size

        in_transformers = transforms.Compose([
        transforms.Resize( (new_size, new_size) ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    content_image = in_transformers(content_image).unsqueeze(0)
    style_image = in_transformers(style_image).unsqueeze(0)

    return content_image, style_image