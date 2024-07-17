import uwimg
import process_image
import os

# 1. Getting and setting pixels
file_name = os.path.join(os.path.dirname(__file__), 'data/dog.jpg')
assert os.path.exists(file_name)

#img = cv2.imread(file_name, -1)
im  = uwimg.Image.load_image(file_name)
for row in range(im.h):
    for col in range(im.w):
        process_image.set_pixel(im, col, row, 0, 0)
im.save_image("dog_no_red")

# # 3. Grayscale image
# im = load_image("data/colorbar.png")
# graybar = rgb_to_grayscale(im)
# save_image(graybar, "graybar")

# # 4. Shift Image
# im = load_image("data/dog.jpg")
# shift_image(im, 0, .4)
# shift_image(im, 1, .4)
# shift_image(im, 2, .4)
# save_image(im, "overflow")

# # 5. Clamp Image
# clamp_image(im)
# save_image(im, "doglight_fixed")

# # 6-7. Colorspace and saturation
# im = load_image("data/dog.jpg")
# rgb_to_hsv(im)
# shift_image(im, 1, .2)
# clamp_image(im)
# hsv_to_rgb(im)
# save_image(im, "dog_saturated")