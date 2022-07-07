#%%
import torch
from torch import nn
from torchgan import layers as gann
from torchinfo import summary as torch_summary
    
from utils import device, image_to_input, ConstrainedConv2d, init_weights, shape_out, reshape_shape, cat_shape



seed_size      = 256
seed_channels  = 64
digit_channels = 64
class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        seed_shape = (1, seed_size)
        self.seed_in = nn.Sequential(
            nn.Linear(seed_size, seed_channels*7*7),
            nn.PReLU())
        seed_shape = shape_out(self.seed_in, seed_shape)
        seed_shape = reshape_shape(seed_shape, (1, seed_channels, 7, 7))
        
        digit_shape = (1, 10)
        self.digit_in = nn.Sequential(
            nn.Linear(10, digit_channels*7*7),
            nn.LeakyReLU())
        digit_shape = shape_out(self.digit_in, digit_shape)
        digit_shape = reshape_shape(digit_shape, (1, digit_channels, 7, 7))
        next_shape = cat_shape(seed_shape, digit_shape, 1)
        
        self.dense_1 = gann.DenseBlock2d(
            depth = 4, 
            in_channels = next_shape[1], 
            growth_rate = 32, 
            block = gann.BasicBlock2d, 
            kernel = 3, 
            stride = 1, 
            padding = 1, 
            batchnorm = False, 
            nonlinearity = None)
        next_shape = shape_out(self.dense_1, next_shape)
        
        self.tcnn_1 = nn.Sequential(
            gann.TransitionBlock2d(
                in_channels = next_shape[1], 
                out_channels = 256, 
                kernel = 3, 
                stride = 1, 
                padding = 1, 
                batchnorm = False, 
                nonlinearity = None),
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True))
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
                out_channels = 256, 
                kernel = 3, 
                stride = 1, 
                padding = 1, 
                batchnorm = False, 
                nonlinearity = None),
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True))
        next_shape = shape_out(self.tcnn_2, next_shape)  
        
        self.attn = gann.SelfAttention2d(
            input_dims = next_shape[1], 
            return_attn=False)
        next_shape = shape_out(self.attn, next_shape)  
        
        self.image_out = nn.Sequential(
            ConstrainedConv2d(
                    in_channels = next_shape[1], 
                    out_channels = 1, 
                    kernel_size = 1),
                nn.Tanh())
        
        self.seed_in.apply(     init_weights).float()
        self.digit_in.apply(    init_weights).float()
        self.dense_1.apply(     init_weights).float()
        self.tcnn_1.apply(      init_weights).float()
        self.dense_2.apply(     init_weights).float()
        self.tcnn_2.apply(      init_weights).float()
        self.attn.apply(        init_weights).float()
        self.image_out.apply(   init_weights).float()
        self.to(device)   
    
    def forward(self, seed, digit):
        seed = seed.to(device)
        digit = digit.to(device)
        seed  = self.seed_in(seed)
        digit = self.digit_in(digit)
        x = torch.cat([seed, digit], -1)
        x = x.reshape(x.shape[0], seed_channels + digit_channels, 7, 7)
        x = self.dense_1(x); x = self.tcnn_1(x)
        x = self.dense_2(x); x = self.tcnn_2(x)
        x = self.attn(x)
        image = self.image_out(x)
        image = image_to_input(image, reverse = True)
        image = image.cpu()
        return(image)
    
if __name__ == "__main__":
    print("\n\n\n")
    gen = Generator()
    print(gen)
    print()
    print(torch_summary(gen, ((1, seed_size), (1,10))))
    
    
# %%
