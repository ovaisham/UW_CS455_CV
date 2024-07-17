import numpy as np
import uwimg
import process_image
from os.path import isfile

"""
Code to create random image
"""
#np.random.seed(21)
#img = uwimg.Image(np.random.rand(500,500,3)*255.0)

img = uwimg.Image.load_image("data/dog.jpg")

# if isfile("0_j.jpg") is None:
#     img.save("0_j.jpg")
# if isfile("0_p.png") is None:
#     img.save("0_p.png")

print(img.h,img.w,img.c)
#process_image.set_pixel(img, 10, 50, 0, val/2)
#print(process_image.get_pixel(img, 10, 50, 0))