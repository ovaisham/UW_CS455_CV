import numpy as np
import uwimg
import process_image
from os.path import isfile

np.random.seed(21)

img = uwimg.Image(np.random.rand(500,500,3)*255.0)

if isfile("0_j.jpg") is None:
    img.save("0_j.jpg")
if isfile("0_p.png") is None:
    img.save("0_p.png")

val = process_image.get_pixel(img, 499, 499, 2)
print(val)
#process_image.set_pixel(img, 10, 50, 0, val/2)
#print(process_image.get_pixel(img, 10, 50, 0))