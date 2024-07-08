import numpy as np
import uwimg
import process_image

img = uwimg.Image(np.random.rand(500,500,3)*255.0)

img.save("0_j.jpg")
img.save("0_p.png")