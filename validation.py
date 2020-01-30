import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color

import utils

# ----------------------------------------
#                 Testing
# ----------------------------------------
def test(opt, rgb, colornet):
    noise = utils.get_noise(1, opt.z_dim, opt.random_type).cuda()
    out_rgb = colornet(rgb, noise)
    out_rgb = out_rgb.cpu().detach().numpy().reshape([3, 256, 256])
    out_rgb = out_rgb.transpose(1, 2, 0)
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb
    
def getImage(root):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    img = Image.open(root).convert('L')
    #img = img.crop((256, 0, 512, 256))
    rgb = img.resize((256, 256), Image.ANTIALIAS)
    rgb = transform(rgb)
    rgb = rgb.reshape([1, 1, 256, 256]).cuda()
    return rgb

def comparison(opt, colornet):
    # Read raw image
    img = Image.open(opt.test_image_name).convert('RGB')
    real = img.crop((0, 0, 256, 256))
    real = real.resize((256, 256), Image.ANTIALIAS)
    real = np.array(real)
    # Forward propagation
    torchimg = getImage(opt.test_image_name)
    out_rgb = test(opt, torchimg, colornet)
    # Show
    out_rgb = np.concatenate((out_rgb, real), axis = 1)
    img_rgb = Image.fromarray(out_rgb)
    img_rgb.show()
    return img_rgb

def colorization(opt, colornet):
    # Forward propagation
    torchimg = getImage(opt.test_image_name)
    out_rgb = test(opt, torchimg, colornet)
    # Show
    img_rgb = Image.fromarray(out_rgb)
    img_rgb.show()
    return img_rgb

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Loading parameters
    parser.add_argument('--pre_train', type = bool, default = False, help = 'pre-train ot not')
    parser.add_argument('--load_name', type = str, default = './models/Pre_only_colorization_epoch10_bs32.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--test_image_name', type = str, default = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256\\ILSVRC2012_val_00008196.JPEG', help = 'test image name')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 1, help = 'grayscale')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'the diverse output RGB images')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Other parameters
    parser.add_argument('--z_dim', type = int, default = 8, help = 'the dimension of adding noise')
    parser.add_argument('--random_type', type = str, default = 'gaussian', help = 'the type of adding noise')
    parser.add_argument('--random_var', type = float, default = 1.0, help = 'the var of adding noise')
    parser.add_argument('--choice', type = str, default = 'colorization', help = 'choice of test operation')
    parser.add_argument('--save', type = bool, default = False, help = 'whether the result image needs to be saved')
    opt = parser.parse_args()

    # Define the basic variables
    colornet = utils.create_generator(opt).cuda()

    # comparison: Compare the colorization output and ground truth
    # colorization: Show the colorization as original size
    if opt.choice == 'comparison':
        img_rgb = comparison(opt, colornet)
        if opt.save:
            imgname = opt.test_image_name.split('/')[-1]
            img_rgb.save('./' + imgname)
    if opt.choice == 'colorization':
        img_rgb = colorization(opt, colornet)
        if opt.save:
            imgname = opt.test_image_name.split('/')[-1]
            img_rgb.save('./' + imgname)
    