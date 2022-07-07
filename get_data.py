#%%
import torch

from keras.datasets import mnist 
import numpy as np
from random import sample

from utils import plot_images

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_y = np.eye(10)[train_y]
test_y  = np.eye(10)[test_y]

train_x = torch.from_numpy(train_x).reshape((train_x.shape[0], 28, 28, 1))
train_y = torch.from_numpy(train_y)
test_x  = torch.from_numpy(test_x).reshape((test_x.shape[0], 28, 28, 1))
test_y  = torch.from_numpy(test_y)

train_x = train_x/255
test_x  = test_x /255

def get_data(batch_size = 64, test = False):
    if(test): x = test_x;  y = test_y
    else:     x = train_x; y = train_y
    index = [i for i in range(len(x))]
    batch_index = sample(index, batch_size)
    x = x[batch_index]
    y = y[batch_index]
    return(x, y.float())

def get_display_data():
    images, digits = get_data(256, True)
    images_for_display = []
    digits_for_display = []
    for i in range(10):
        for j, d in enumerate(digits):
            if(d.argmax().item() == i):
                images_for_display.append(images[j].unsqueeze(0))
                digits_for_display.append(d.unsqueeze(0))
                break 
    return(
        torch.cat(images_for_display).cpu(), 
        torch.cat(digits_for_display))

if __name__ == "__main__":
    x, y = get_display_data()
    plot_images(x, "Real numbers", y.argmax(-1)) 
# %%
