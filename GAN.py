import torch
from torch import nn
from torch.optim import Adam
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss

import os 
from PIL import Image

from utils import plot_losses, plot_images, plot_acc, delete_these, save_models, load_models, \
    start_time, duration
from get_data import get_data, get_display_data
from generator import Generator, seed_size
from discriminator import Discriminator



class GAN:
    def __init__(self, load = "", d = 3):
        self.start = start_time()
        self.gen_lr = .0001
        self.dis_lr = .0004
        self.betas = (0.5, 0.999)
        
        self.gen = Generator()
        self.gen_opt  = Adam(self.gen.parameters(), self.gen_lr, self.betas)

        self.d = d
        self.dis = [Discriminator() for _ in range(d)]
        self.dis_opts = [Adam(dis.parameters(), self.dis_lr, self.betas) for dis in self.dis]
        
        if(load != ""): self.gen, self.dis = load_models(self.gen, self.dis, load)
        
        self.bce = nn.BCELoss()
        # These losses currently not used
        self.gen_loss = WassersteinGeneratorLoss()
        self.dis_loss = WassersteinDiscriminatorLoss()
        
        self.display_images, self.display_digits = get_display_data()
        self.display_seeds = self.get_seeds(10)
        
        self.gen_train_losses = []; self.gen_test_losses  = []
        self.dis_train_losses = [[] for _ in range(d)]; self.dis_test_losses  = [[] for _ in range(d)]
        self.train_fakes_acc =  [[] for _ in range(d)]; self.test_fakes_acc =   [[] for _ in range(d)]
        self.train_reals_acc =  [[] for _ in range(d)]; self.test_reals_acc =   [[] for _ in range(d)]
        
    def get_seeds(self, batch_size):
        return(torch.randn(size = (batch_size, seed_size)))
    
    
    
    def make_train(self):
        self.gen.train() 
        for dis in self.dis: dis.train() 
    
    def make_eval(self):
        self.gen.eval() 
        for dis in self.dis: dis.eval() 
            
            

    def epoch(self, batch_size, train, verbose = False):
        if(not train): self.make_eval()
        else:          self.make_train()
        
        images, digits = get_data(batch_size, train)
        seeds = self.get_seeds(batch_size)
        gen_images = self.gen(seeds, digits.detach())
        
        image_noise     = torch.normal(torch.zeros(images.shape), .1*torch.ones(images.shape))
        gen_image_noise = torch.normal(torch.zeros(images.shape), .1*torch.ones(images.shape))
        images     = images + image_noise 
        gen_images = gen_images + gen_image_noise
                
        for d in range(self.d):
            self.dis[d].zero_grad()
            judgement = self.dis[d](images.detach(), digits.detach()).squeeze(-1)
            correct = .9*torch.ones(judgement.shape)
            real_loss = self.bce(judgement, correct)
            reals_correct = [1 if round(judgement[i].item()) == round(correct[i].item()) else 0 for i in range(len(judgement))]        
            reals_accuracy = sum(reals_correct)/len(reals_correct)
            if(train): real_loss.backward()
            
            judgement = self.dis[d](gen_images.detach(), digits.detach()).squeeze(-1)
            correct = correct.fill_(0)
            fake_loss = self.bce(judgement, correct)
            fakes_correct = [1 if round(judgement[i].item()) == round(correct[i].item()) else 0 for i in range(len(judgement))]
            fakes_accuracy = sum(fakes_correct)/len(fakes_correct)
            if(train): fake_loss.backward()
            
            loss = real_loss + fake_loss 
            loss = loss.cpu().detach()

            if(train):
                self.dis_opts[d].step()
                self.dis_train_losses[d].append(loss)
                self.train_reals_acc[d].append(reals_accuracy)
                self.train_fakes_acc[d].append(fakes_accuracy)
            else:
                self.dis_test_losses[d].append(loss)
                self.test_reals_acc[d].append(reals_accuracy)
                self.test_fakes_acc[d].append(fakes_accuracy)
    
        self.gen.zero_grad()
        seeds = self.get_seeds(batch_size)
        gen_images = self.gen(seeds, digits.detach())
        judgements = []
        for d in range(self.d):
            judgements.append(self.dis[d](gen_images, digits))
        judgements = torch.cat(judgements, -1)
        loss = self.bce(judgements, torch.ones(judgements.shape))
        
        if(train):
            loss.backward()
            self.gen_opt.step()
            self.gen_train_losses.append(loss.cpu().detach())
        else:
            self.gen_test_losses.append(loss.cpu().detach())
            
        delete_these(verbose, images, gen_images, digits, seeds, judgement)
            
            
                
    def train(self, epochs = 100, batch_size = 64, announce = 10, save = 10, display = 100):
        for e in range(epochs):
            self.e = e
            if(e%announce == 0 or e == 0):
                print("Epoch {}, {}. ".format(e, duration(self.start)), end = "")
            if(e%save == 0 or e == 0):
                self.display(show = False, name = "epoch_{}".format(str(e).zfill(6)))
            if(e%display == 0 or e == 0):
                self.display()
                
            self.epoch(batch_size, train = True)
            self.epoch(batch_size, train = False)
            


    def display(self, show = True, name = ""):
        plot_losses(self.gen_train_losses, self.gen_test_losses, 
                    "Generator Losses : Epoch {}".format(self.e), show = show, name = name, folder = "gen_loss")
        for d in range(self.d):
            plot_losses(self.dis_train_losses[d], self.dis_test_losses[d], 
                        "Discriminator {} Losses : Epoch {}".format(d, self.e), show = show, name = name if name == "" else name + "_{}".format(d), folder = "dis_loss")
            plot_acc(
                self.train_fakes_acc[d], self.train_reals_acc[d], 
                self.test_fakes_acc[d],  self.test_reals_acc[d], 
                "Discriminator {} Accuracy : Epoch {}".format(d, self.e), show = show, name = name if name == "" else name + "_{}".format(d), folder = "accuracy")
        plot_images(images = self.display_images, title = "Real images",
                    titles = self.display_digits.argmax(-1), 
                    rows = 2, columns = 5, show = show, name = "")
        plot_images(images = self.gen(self.display_seeds, self.display_digits).cpu().detach(), 
                    title = "Fakes : Epoch {}".format(self.e), titles = self.display_digits.argmax(-1), 
                    rows = 2, columns = 5, show = show, name = name, folder = "fakes")
        save_models(self.gen, self.dis, name = name)
        
        
    def animate(self, num = 5, frames_between = 5, duration = 1):
        for f in os.listdir("images/animation"):
            os.remove("images/animation/{}".format(f))
        
        seeds = self.get_seeds(num)
        seeds = torch.cat([seeds, seeds[0].unsqueeze(0)])
        frame = 0
        for seed_1, seed_2 in zip(seeds[:-1], seeds[1:]):
            for i in range(frames_between):
                ratio = i/frames_between 
                between_seed = seed_1*(1-ratio) + seed_2*ratio
                between_seeds = between_seed.unsqueeze(0).tile((10,1))
                plot_images(images = self.gen(between_seeds, self.display_digits).cpu().detach(), 
                    title = "Animation", titles = self.display_digits.argmax(-1), 
                    rows = 2, columns = 5, show = False, name = str(frame).zfill(6), folder = "animation")
                frame += 1
                
        files = []
        for file in os.listdir("images/animation"):
            files.append(file)
        files.sort()
        frames = []
        for file in files:
            new_frame = Image.open("images/animation/"+file)
            frames.append(new_frame)
        frames[0].save('images/animation/animation.gif', format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=duration, loop=0)