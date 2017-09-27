from PIL import Image
import os, errno
import image_slicer
from shutil import move

new_width, new_height = 1280, 720
number_of_slices = 4
sourceDirectory='/Users/kris/Downloads/football1'
destinationDirectory='/Users/kris/Downloads/football2-resized2'

if not os.path.exists(destinationDirectory):
    os.makedirs(destinationDirectory)

# Slice images one by one and copy them to the new directory
for filename in os.listdir(sourceDirectory):
    if filename.endswith(".jpg"):
        print("Slicing: {0}".format(os.path.join(sourceDirectory, filename)))
        pre, ext = os.path.splitext(filename)
        tiles = image_slicer.slice(os.path.join(sourceDirectory, filename), number_of_slices, save=False)
        image_slicer.save_tiles(tiles, directory=destinationDirectory, prefix=pre)


for filename in os.listdir(destinationDirectory):
    if filename.endswith(".png"):
        print("Resizing: {0}".format(os.path.join(destinationDirectory, filename)))
        img = Image.open(os.path.join(destinationDirectory, filename))
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(os.path.join(destinationDirectory, filename))
        continue
    else:
        continue

print('Done!')
