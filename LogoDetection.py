# This might be run only once if no ffmpeg is installed
#import imageio
#imageio.plugins.ffmpeg.download()

import cv2
import numpy as np
import os
import time
from PIL import Image
from math import sqrt, ceil, floor
import numpy

from google.protobuf import text_format
from moviepy.editor import VideoFileClip
import image_slicer

import scipy.misc
os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output

import caffe
from caffe.proto import caffe_pb2

countedLogos = []
frameNumber = 0
foundLogos = False

DEPLOY_FILE = '/Users/kris/Downloads/fedex_mil_model_epoch_70.0/deploy.prototxt'
MEAN_FILE = None #'/Users/kris/Downloads/fedex_mil_model_epoch_70.0/mean.binaryproto'
MODEL_FILE = '/Users/kris/Downloads/fedex_mil_model_epoch_70.0/snapshot_iter_28000.caffemodel'
BATCH_SIZE = 1
OUTPUT_FILE = 'output.mp4'
INPUT_FILE = '/Users/kris/Downloads/footbal-split-movie-57s.mp4'
USE_GPU = False

# IMAGE_FILE = '/Users/kris/Downloads/football2-resized2/resized/195583_01_01.png'

class Logo:
    """Represents a single logo set."""
    frameNumber = 0
    fedexCounter = 0
    def __init__(self, frameNumber, fedexCounter):
        self.frameNumber = frameNumber
        self.fedexCounter = fedexCounter



def get_net(caffemodel, deploy_file, use_gpu=False):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        println("mean file exists")
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def resize_img(image, height, width):
    """
    Resizes the image to detectnet inputs

    Arguments:
    image -- a single image
    height -- height of the network input
    width -- width of the network input
    """
    image = np.array(image)
    image = scipy.misc.imresize(image, (height, width), 'bilinear')

    #image = image.resize((width, height), Image.ANTIALIAS)
    return image

def draw_bboxes(image, locations):
    """
    Draws the bounding boxes into an image

    Arguments:
    image -- a single image already resized
    locations -- the location of the bounding boxes
    """
    for left,top,right,bottom,confidence in locations:
        if confidence==0:
            continue
        print("confidence > 0")
        foundLogos = True
        cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),3)
    #cv2.imwrite('bbox.png',image)#test on a single image
    return image

def forward_pass(image, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    image -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []

    if image.ndim == 2:
        caffe_images.append(image[:,:,np.newaxis])
    else:
        caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores

def classify(caffemodel, deploy_file, image,
        mean_file=None, batch_size=None, use_gpu=False):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)

    image = resize_img(image, height, width)

    # Classify the image
    scores = forward_pass(image, net, transformer, batch_size=batch_size)
    # print("Scores: ")
    # print(scores)
    ### Process the results

    # Format of scores is [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    # https://github.com/NVIDIA/caffe/blob/v0.15.13/python/caffe/layers/detectnet/clustering.py#L81
    for i, image_results in enumerate(scores):
        print '==> Image #%d' % i
        img_result = draw_bboxes(image,image_results)
    # This line is optinal, in this case we resize to the size of the original input video, can be removed
    #img_result = resize_img(img_result,720,1280)
    return img_result

def detect_logos(image):
    """
    Runs our pipeline given a single image and returns another one with the bounding boxes drawn

    Arguments:
    image -- cv2 image file being 1/4 of the original
    """
    result = classify(MODEL_FILE, DEPLOY_FILE, image, MEAN_FILE, BATCH_SIZE, USE_GPU)
    return result

def slice_image(image, number_tiles):
    im = toPILImage(image)
    im_w, im_h = im.size
    columns, rows = image_slicer.calc_columns_rows(number_tiles)
    extras = (columns * rows) - number_tiles
    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h): # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w): # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1,
                        int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = image_slicer.Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    return tuple(tiles)

def toPILImage(opencvImage):
    img = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    return im

def toOpenCVFormat(pilImage):
    open_cv_image = numpy.array(pilImage)
    image = open_cv_image[:, :, ::-1].copy()
    return image

def joinImages(modified_images):
    result = Image.new("RGB", (1280, 720))

    for index, img in enumerate(modified_images):
        img.thumbnail((640, 360), Image.ANTIALIAS)
    result.paste(modified_images[0], (0, 0))
    result.paste(modified_images[2], (0, 360))
    result.paste(modified_images[1], (640, 0))
    result.paste(modified_images[3], (640, 360))
    return result

def detect_logos_full_img(image):
    """
    Splits a single frame into 4 images

    Arguments:
    image -- cv2 image file for single video frame
    """
    tiles = slice_image(image, 4)
    modified_images = []
    print("tiles length: " + str(len(tiles)))
    for i, tile in enumerate(tiles):
        open_cv_image = numpy.array(tile.image)
        image = open_cv_image[:, :, ::-1].copy()
        img_detected = detect_logos(image)
        modified_images.append(toPILImage(img_detected))
    img = joinImages(modified_images)

    # if foundLogos:
        # countedLogos.append(Logo(frameNumber, 1))
    # frameNumber += 1
    # foundLogos = False
    return toOpenCVFormat(img)

if __name__ == '__main__':
    script_start_time = time.time()

    project_output = OUTPUT_FILE

    clip1 = VideoFileClip(INPUT_FILE);
    white_clip = clip1.fl_image(detect_logos_full_img)
    white_clip.write_videofile(project_output, audio=True);

    # IMAGE_FILE = '/Users/kris/Downloads/football2-resized2/resized/184196_01_01.png'
    # image = cv2.imread(IMAGE_FILE);
    # img = detect_logos_full_img(image)
    # cv2.imwrite('frame.jpg',img)
    print 'Video took %f seconds.' % (time.time() - script_start_time)
    print 'Counted logos: ' + str(countedLogos)
