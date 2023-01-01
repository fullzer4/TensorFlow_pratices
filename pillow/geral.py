from PIL import Image

my_image = "../testImgs/lenna.png"

image = Image.open(my_image)
print(type(image))