#%%
import os

if not os.path.exists('images'):
    os.makedirs('images')

folders = ["gen_loss", "dis_loss", "accuracy", "fakes", "gen", "dis", "animation"]

for folder in folders:
    if not os.path.exists('images/{}'.format(folder)):
        os.makedirs('images/{}'.format(folder))

for folder in folders:
    for f in os.listdir("images/{}".format(folder)):
        os.remove("images/{}/{}".format(folder, f))
        
#%%

from GAN import GAN

gan = GAN(d = 3)
gan.train(epochs = 2000+1, batch_size = 128)

# %%

from GAN import GAN

files = os.listdir("images/gen")
files.sort()
gan = GAN(load = files[-1][:-3], d = 3)
gan.animate(num = 10, frames_between = 20, duration = 5)

# %%
from PIL import Image

files = []
for file in os.listdir("images/fakes"):
    files.append(file)
files.sort()
frames = []
for file in files:
    new_frame = Image.open("images/fakes/"+file)
    frames.append(new_frame)
frames[0].save('images/fakes/animation.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=3, loop=0)