import os, errno
import xml.etree.ElementTree
import random
from sklearn.cross_validation import train_test_split
import numpy
from shutil import copyfile
import shutil



new_width, new_height = 1248, 384
sourceDirectory='/Users/kris/Downloads/football1-resized'
destinationDirectory='/Users/kris/Downloads/football1-resized2'

# remove the dir First
if os.path.exists(destinationDirectory):
    shutil.rmtree(destinationDirectory)

if not os.path.exists(destinationDirectory):
    os.makedirs(destinationDirectory)
    os.makedirs(destinationDirectory + '/train')
    os.makedirs(destinationDirectory + '/train/images')
    os.makedirs(destinationDirectory + '/train/labels')
    os.makedirs(destinationDirectory + '/val')
    os.makedirs(destinationDirectory + '/val/images')
    os.makedirs(destinationDirectory + '/val/labels')

def split():
    data = numpy.array(os.listdir(sourceDirectory + '/annotations'))
    train ,test = train_test_split(data,test_size=0.2)
    return train, test

def processSingleFile(filename):
    lines = []
    if filename.endswith(".xml"):
        print("Processing: {0}".format(os.path.join(sourceDirectory + '/annotations', filename)))
        e = xml.etree.ElementTree.parse(os.path.join(sourceDirectory + '/annotations', filename)).getroot()
        name, xmin, ymin, xmax, ymax = '','','','',''
        for elem in e.iterfind('object'):
            for oel in elem.iter():
                if(oel.tag == 'name'):
                    name = oel.text.strip()
                elif(oel.tag == 'xmin'):
                    xmin = oel.text.strip()
                elif(oel.tag == 'ymin'):
                    ymin = oel.text.strip()
                elif(oel.tag == 'xmax'):
                    xmax = oel.text.strip()
                elif(oel.tag == 'ymax'):
                    ymax = oel.text.strip()
                else:
                    continue
            # Example: Car   0.00 0 3.11 801.18 169.78 848.06 186.03 1.41 1.59 4.01 19.44 1.17 65.54 -2.89
            # name 0.00 0 0.00 xmin ymax xmax ymin 0.00 0.00 0.00 0.00 0.00 0.00 0.00
            lines.append(name + ' ' + '0.00 0 0.00 ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' 0.00 0.00 0.00 0.00 0.00 0.00 0.00')
    return lines


def convertToKitti(labelsDir, imagesDir, files):
    for f in files:
        pre, ext = os.path.splitext(f)
        # create label file
        lines = processSingleFile(f)
        kitti_labels = pre + '.txt'
        with open(os.path.join(destinationDirectory + labelsDir, kitti_labels), "w") as text_file:
            for item in lines:
                text_file.write("%s\n" % item)
        # copy the image file
        imgFile = pre + '.jpg'
        srcFile = os.path.join(sourceDirectory, imgFile)
        dstFile = os.path.join(destinationDirectory + imagesDir, imgFile)
        copyfile(srcFile, dstFile)

print("splitting into traint and test sets: ")
train, test = split()

# Convert test subset
convertToKitti('/val/labels', '/val/images', test)

# Convert train subset
convertToKitti('/train/labels', '/train/images', train)

print('Done!')
