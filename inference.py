from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import torch
import torch.optim as optim
from torchvision import transforms, models

class Master: 

    def __init__(self):
        # get the "features" portion of VGG19 (we will not need the "classifier" portion)
        self.vgg = models.vgg19(torch.load('vgg19-dcbb9e9d.pth')).features
        # freeze all VGG parameters since we're only optimizing the target image
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        # move the model to GPU, if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg.to(self.device)
    
    
    def load_image(self,img_path, max_size=400, shape=None):
        ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
        image = Image.open(img_path).convert('RGB')
    
        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
        
        if shape is not None:
            size = shape
        
        in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
    
        return image
    
# content and style image sequezz
    def process_input(self,main_addr,style_addr):
        
        content = self.load_image(main_addr).to(self.device)
        # Resize style to match content, makes code easier
        style = self.load_image(style_addr, shape=content.shape[-2:]).to(self.device) 

        return content,style
    

    def im_convert(self,tensor):
        """ Display a tensor as an image. """
        
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image
    

    def get_features(self,image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1', 
                      '10': 'conv3_1', 
                      '19': 'conv4_1',
                      '21': 'conv4_2',  ## content representation
                      '28': 'conv5_1'}
        
        features = {}
        x = image
        
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        
        return features
    
    def gram_matrix(self,tensor):
        # get the batch_size, depth, height, and width of the Tensor
        _, d, h, w = tensor.size()
    
        # reshape so we're multiplying the features for each channel
        tensor = tensor.view(d, h * w)
    
        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
    
        return gram 
    
    def put_to(self,addr1,addr2):# get content and style features only once before training

        content,style = self.process_input(addr1,addr2)
        content_features = self.get_features(content, self.vgg)
        style_features = self.get_features(style, self.vgg)

        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        target = content.clone().requires_grad_(True).to(self.device)

        style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}
        
        content_weight = 1  # alpha
        style_weight = 1e9  # beta

        show_every = 400
        
        optimizer = optim.Adam([target], lr=0.003)
        steps = 2000  # decide how many iterations to update your image (5000)

        for ii in range(1, steps+1):
            
            target_features = self.get_features(target, self.vgg)
            
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
            style_loss = 0
            
            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                
                style_gram = style_grams[layer]
                
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                
                style_loss += layer_style_loss / (d * h * w)
            
            total_loss = content_weight * content_loss + style_weight * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # display intermediate images and print the loss
            """ if  ii % show_every == 0:
                print('Total loss: ', total_loss.item())
                plt.imshow(self.im_convert(target))
                plt.show()
                """
        mpimg.imsave("Content.png", self.im_convert(content))
        mpimg.imsave("style.png", self.im_convert(target))


m = Master()
m.put_to("data/input_image/RKS.jpg","data/target_style/acrylic.jpg")

    
