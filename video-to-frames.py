import cv2
import math


print(cv2.__version__)
vidcap = cv2.VideoCapture('UEL.2017.09.14.Hoffenheim.vs.Braga.EN.720p-FS.mp4')
frameRate = vidcap.get(5)
print 'Frame rate: ', frameRate
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  frameId = vidcap.get(1) #current frame number
  if (frameId % math.floor(frameRate) == 0):
      print 'Read a new frame: ', success
      cv2.imwrite("/Users/kris/Downloads/football1/%d.jpg" % count, image)     # save frame as JPEG file
  count += 1

vidcap.release()
print "Done!"
