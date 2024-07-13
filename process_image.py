import numpy as np
from uwimg import Image

# class Image:
#     def __init__(self, width, height, channels):
#         self.w = width
#         self.h = height
#         self.c = channels
#         self.data = np.zeros((height, width, channels), dtype=np.float32)

def get_pixel(im: Image, x: int, y:int, c:int) -> float:
    # x is column, y is row, c is channel
    data = im.data
    if x<=0:
        x=0
    elif x>=(im.w-1):
        x=(im.w-1)
    if y<=0:
        y=0
    elif y>=(im.h-1):
        y=im.h-1
    if c<=0:
        c=0
    elif c>=(im.c-1):
        c=im.c-1
    try:
        return data[y, x, c]
    except IndexError as i:
        return f"{i.args[0]}"
    except TypeError as t:
        return f"{t.args[0]}"
        # x<0 or x>=len(data[:,0,0]) or y<0 or y>=len(data[0,:,0]) or c<0:

def set_pixel(im: Image, x: int, y:int, c:int, v:float):
    
    try:
        im.data[y, x, c]= v
    except:
        pass

def copy_image(im: Image):
    copy = Image(im.w, im.h, im.c)
    # TODO Fill this in
    return copy

def rgb_to_grayscale(im):
    assert im.c == 3
    gray = Image(im.w, im.h, 1)
    # TODO Fill this in
    return gray

def shift_image(im, c, v):
    # TODO Fill this in
    pass

def clamp_image(im):
    # TODO Fill this in
    pass

def three_way_max(a, b, c):
    return max(a, b, c)

def three_way_min(a, b, c):
    return min(a, b, c)

def rgb_to_hsv(im):
    # TODO Fill this in
    pass

def hsv_to_rgb(im):
    # TODO Fill this in
    pass
