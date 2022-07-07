#%%
import torch
from torch import nn
from torch import linalg as LA
from torchgan import layers as gann
from torchinfo import summary as torch_summary
    
from utils import device, image_to_input, ConstrainedConv2d, init_weights, \
    shape_out, flatten_shape, cat_shape, reshape_shape

    
    
norm_size       = 64
digits_channels = 64
class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
             
        image_shape = (1, 1, 28, 28)   
        self.dense_1 = gann.DenseBlock2d(
            depth = 4, 
            in_channels = image_shape[1], 
            growth_rate = 32, 
            block = gann.BasicBlock2d, 
            kernel = 3, 
            stride = 1, 
            padding = 1, 
            batchnorm = False, 
            nonlinearity = None)
        next_shape = shape_out(self.dense_1, image_shape)
        
        self.tcnn_1 = nn.Sequential(
            gann.TransitionBlock2d(
                in_channels = next_shape[1], 
                out_channels = 128, 
                kernel = 3, 
                stride = 1, 
                padding = 1, 
                batchnorm = False, 
                nonlinearity = None),
            nn.Upsample(scale_factor = .5, mode = "bilinear", align_corners = True))
        next_shape = shape_out(self.tcnn_1, next_shape)   
        
        self.dense_2 = gann.DenseBlock2d(
            depth = 4, 
            in_channels = next_shape[1], 
            growth_rate = 32, 
            block = gann.BasicBlock2d, 
            kernel = 3, 
            stride = 1, 
            padding = 1, 
            batchnorm = False, 
            nonlinearity = None)
        next_shape = shape_out(self.dense_2, next_shape)
        
        self.tcnn_2 = nn.Sequential(
            gann.TransitionBlock2d(
                in_channels = next_shape[1], 
                out_channels = 128, 
                kernel = 3, 
                stride = 1, 
                padding = 1, 
                batchnorm = False, 
                nonlinearity = None),
            nn.Upsample(scale_factor = .5, mode = "bilinear", align_corners = True))
        next_shape = shape_out(self.tcnn_2, next_shape)    
        
        digit_shape = (1, 10)
        self.digit_in = nn.Sequential(
            nn.Linear(10, digits_channels*7*7),
            nn.PReLU())     
        digit_shape = shape_out(self.digit_in, digit_shape)
        digit_shape = reshape_shape(digit_shape, (1, digits_channels, 7, 7))
        
        self.digit_cnn = nn.Sequential(
            ConstrainedConv2d(
                in_channels = digit_shape[1],
                out_channels = digits_channels, 
                kernel_size = 3,
                padding = (1,1),
                padding_mode = "reflect"),
            nn.PReLU())
        digit_shape = shape_out(self.digit_cnn, digit_shape)
        next_shape = cat_shape(next_shape, digit_shape, 1)
        
        self.dense_3 = nn.Sequential(
            nn.Dropout(.2),
            gann.DenseBlock2d(
                depth = 4, 
                in_channels = next_shape[1], 
                growth_rate = 32, 
                block = gann.BasicBlock2d, 
                kernel = 3, 
                stride = 1, 
                padding = 1, 
                batchnorm = False, 
                nonlinearity = None))
        next_shape = shape_out(self.dense_3, next_shape)
        
        self.tcnn_3 = nn.Sequential(
            nn.Dropout(.2),
            gann.TransitionBlock2d(
                in_channels = next_shape[1], 
                out_channels = 128, 
                kernel = 3, 
                stride = 1, 
                padding = 1, 
                batchnorm = False, 
                nonlinearity = None))
        next_shape = shape_out(self.tcnn_3, next_shape)   
        
        self.attn = nn.Sequential(
            nn.Dropout(.2),
            gann.SelfAttention2d(
                input_dims = next_shape[1], 
                return_attn = False))
        next_shape = shape_out(self.attn, next_shape)  
        next_shape = flatten_shape(next_shape, 1)
        
        norm_shape = (1, 1)
        self.norm_in = nn.Sequential(
            nn.Linear(1, norm_size),
            nn.PReLU())
        norm_shape = shape_out(self.norm_in, norm_shape)
                                    
        self.guess = nn.Sequential(
            nn.Linear(next_shape[1] + norm_shape[1], 1),
            nn.Tanh())
        
        self.dense_1.apply(     init_weights).float()
        self.tcnn_1.apply(      init_weights).float()   
        self.dense_2.apply(     init_weights).float()
        self.tcnn_2.apply(      init_weights).float()   
        self.digit_in.apply(    init_weights).float()   
        self.digit_cnn.apply(   init_weights).float()   
        self.dense_3.apply(     init_weights).float()
        self.tcnn_3.apply(      init_weights).float()   
        self.attn.apply(        init_weights).float()
        self.norm_in.apply(     init_weights).float()   
        self.guess.apply(       init_weights).float()
        self.to(device)
        
    def forward(self, image, digit):
        image = image.to(device)
        digit = digit.to(device)
        image = image_to_input(image, reverse = False)
        norm = LA.norm(image, dim=(1,2,3))
        norm = self.norm_in(norm.unsqueeze(1))
        
        image = self.dense_1(image); image = self.tcnn_1(image)
        image = self.dense_2(image); image = self.tcnn_2(image)
        
        digit = self.digit_in(digit)
        digit = digit.reshape(digit.shape[0], digits_channels, 7, 7)
        digit = self.digit_cnn(digit)
        
        image = torch.cat([image, digit], 1)
        image = self.dense_3(image); image = self.tcnn_3(image)
        image = self.attn(image)
        image = image.flatten(1)
        
        x = torch.cat([image, norm], -1)
        x = (self.guess(x) + 1)/2
        x = x.cpu()
        return(x)
        
if __name__ == "__main__":
    print("\n\n\n")
    dis = Discriminator()
    print(dis)
    print()
    print(torch_summary(dis, ((1, 28, 28, 1), (1,10))))
    
    


# %%
