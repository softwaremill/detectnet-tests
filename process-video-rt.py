import numpy as np
import cv2
import caffe


MODEL_FILE = '/Users/kris/Downloads/fedex_mil_model_epoch_70.0/deploy.prototxt'
PRETRAINED = '/Users/kris/Downloads/fedex_mil_model_epoch_70.0/snapshot_iter_28000.caffemodel'
IMAGE_FILE = '/Users/kris/Downloads/football2-resized2/resized/195583_01_01.png'

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/Users/kris/Downloads/output.mp4')

caffe.set_mode_cpu()

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST);

img = cv2.imread(IMAGE_FILE);

# print(res)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
