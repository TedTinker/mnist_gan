#%%
### A few utilities
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")



from torch import nn
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
def shape_out(layer, shape_in):
    example = torch.zeros(shape_in)
    example = layer(example)
    return(example.shape)

from math import prod
def flatten_shape(shape, num):
    new_shape = tuple(s for i,s in enumerate(shape) if i < num)
    new_shape += (prod(shape[num:]),)
    return(new_shape)

def cat_shape(shape_1, shape_2, dim):
    assert(len(shape_1) == len(shape_2))
    new_shape = ()
    for (s1, s2, d) in zip(shape_1, shape_2, range(len(shape_1))):
        if(d != dim): 
            assert(s1 == s2)
            new_shape += (s1,)
        else:
            new_shape += (s1+s2,)
    return(new_shape)

def reshape_shape(shape, new_shape):
    assert(prod(shape) == prod(new_shape))
    return(new_shape)



# Monitor GPU memory.
def get_free_mem(string = ""):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n{}: {}.\n".format(string, f))

# Remove from GPU memory.
def delete_these(verbose = False, *args):
    if(verbose): get_free_mem("Before deleting")
    del args
    torch.cuda.empty_cache()
    if(verbose): get_free_mem("After deleting")
    
    
    
def image_to_input(image, reverse):
    if(reverse):
        image = image.permute(0, 2, 3, 1)
        image = (image + 1)/2
    else:
        image = image.permute(0, -1, 1, 2)
        image = (image * 2) - 1
    return(image)



def plot_losses(train_losses, test_losses, title, show = True, name = "", folder = ""):
    plt.plot(train_losses, label = "training", color = (1, 0, 0, .5))
    plt.plot(test_losses,  label = "testing",  color = (1, .5, .5, .5))
    plt.legend(loc = 'upper left')
    plt.title(title)
    if(name != ""): plt.savefig("images/{}/{}.png".format(folder, name))
    if(show): plt.show()
    plt.close()
    
def plot_acc(train_fakes_acc, train_reals_acc, test_fakes_acc, test_reals_acc, title, show = True, name = "", folder = ""):
    plt.plot(train_fakes_acc, label = "train fake acc", color = (1, 0, 0, .5))
    plt.plot(test_fakes_acc,  label = "test  fake acc", color = (1, .5, .5, .5))
    plt.plot(train_reals_acc, label = "train real acc", color = (0, 0, 1, .5))
    plt.plot(test_reals_acc,  label = "test  real acc", color = (.5, .5, 1, .5))
    plt.ylim([0,1])
    plt.legend(loc = 'upper left')
    plt.title(title)
    if(name != ""): plt.savefig("images/{}/{}.png".format(folder, name))
    if(show): plt.show()
    plt.close()
    
def plot_image(image, title = "", show = True, name = "", folder = ""):
    plt.imshow(image.squeeze(-1), cmap = "gray")
    plt.axis("off")
    plt.title(title)
    if(name != ""): plt.savefig("images/{}/{}.png".format(folder, name))
    if(show): plt.show()
    plt.close()
    
def plot_images(images, title, titles, rows = 2, columns = 5, show = True, name = "", folder = ""):
    fig = plt.figure(figsize=(columns+1, rows+1))
    fig.suptitle(title)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1].squeeze(-1), cmap = "gray")
        plt.title(titles[i-1].item())
        plt.axis("off")
    if(name != ""): plt.savefig("images/{}/{}.png".format(folder, name))
    if(show): plt.show()
    plt.close()
    
    
    
def save_models(gen, dis_list, name = ""):
    if(name != ""): 
        torch.save(gen.state_dict(), "images/gen/{}.pt".format(name))
        for d, dis in enumerate(dis_list):
            torch.save(dis.state_dict(), "images/dis/{}.pt".format(name + "_{}".format(d)))
            
def load_models(gen, dis_list, name = ""):
    if(name != ""): 
        gen.load_state_dict(torch.load("images/gen/{}.pt".format(name)))
        for d, dis in enumerate(dis_list):
            dis.load_state_dict(torch.load("images/dis/{}.pt".format(name + "_{}".format(d))))
    return(gen, dis_list)



import datetime
def start_time():
    return(datetime.datetime.now())
def duration(start):
    change_time = datetime.datetime.now() - start
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)



# %%
