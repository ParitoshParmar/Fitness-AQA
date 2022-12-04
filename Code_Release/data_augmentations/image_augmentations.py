# Author: Paritosh Parmar

'''
Set parameters according to your needs
'''

from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
from opts_exercise_qa import randomseed
from itertools import product

random.seed(randomseed)


def hori_flip(im):
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im
  

def masking(im, type='fixed', mask_amt=0):
    draw = ImageDraw.Draw(im)
    # draw random sized rectangle in the center of the image
    width, height = im.size
    if type == 'random':
        mask_amt = random.uniform(0, mask_amt)
    mask_height = int(height*mask_amt)
    # upper image masking
    draw.rectangle([(0, 0), (width, mask_height)], fill=(0,0,0))
    # draw.rectangle([(0, 0), (width, mask_height)], fill=(int(0.485*255), int(0.456*255), int(0.406*255)))
    # lower image masking
    # draw.rectangle([(0, mask_height), (width, height)], fill=(int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
    return im


# random checkered pattern masking with possible overlapping squares
def masking_checker_ol(im):
    draw = ImageDraw.Draw(im)

    # x_coords = [9 * i + x for i, x in enumerate(sorted(random.sample(range(), 4)))]
    rect2draw = random.randint(6,8)
    rect_size_ll = 50#int(im.size[0] / 6)
    rect_size_ul = 60#int(im.size[0] / 5)
    x_coords = random.sample(range(320-rect_size_ul),rect2draw)
    y_corrds = random.sample(range(320-rect_size_ul),rect2draw)
    for i in range(rect2draw):
        side = random.randint(rect_size_ll,rect_size_ul)
        draw.rectangle([(x_coords[i], y_corrds[i]), (x_coords[i]+side, y_corrds[i]+side)], fill=(0,0,0))
        # draw.rectangle([(x_coords[i], y_corrds[i]), (x_coords[i] + side, y_corrds[i] + side)],
        #                fill=(int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
    return im


# random checkered pattern masking with non-overlapping squares
def masking_checker_nool(im):
    draw = ImageDraw.Draw(im)
    rect_size_ll = 50#int(im.size[0] / 6)
    rect_size_ul = 60#int(im.size[0] / 5)
    rect2draw = random.randint(3,5)#(6,8)
    a = np.linspace(0, 0.75*im.size[0]+random.randint(0,0.2*im.size[0]), 5, dtype=int)
    b = np.linspace(0, 0.75*im.size[0]+random.randint(0,0.2*im.size[0]), 5, dtype=int)
    c = list(product(a, b))
    random.shuffle(c)
    c = c[:rect2draw]
    for i in range(len(c)):
        side = random.randint(rect_size_ll,rect_size_ul)
        # draw.rectangle([(c[i][0], c[i][1]), (c[i][0] + side, c[i][1] + side)], fill=(0,0,0))
        draw.rectangle([(c[i][0], c[i][1]), (c[i][0] + side, c[i][1] + side)],
                       fill=(int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
    return im
