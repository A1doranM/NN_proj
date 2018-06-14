import cv2
from PIL import Image

#Get images from video
vidcap = cv2.VideoCapture('Original videos\\blue_yellow.mp4')
success, image = vidcap.read()
count = 1000
success = True
while success:
    success, image = vidcap.read()
    print('Read a new frame:', count, " ", success)
    cv2.imwrite('Validation\\Blue\\blue%d.jpg' % count, image)  # save frame as JPEG file
    count += 1

#Resize images
# width = 150
# height = 150
# count = 0
# while count < 465:
#     img = Image.open('FalseCup\\Ohne\\ohne%d.jpg' % count)
#     #ratio = (width / float(img.size[0]))
#     #height = int((float(img.size[1]) * float(ratio)))
#     img = img.resize((width, height), Image.ANTIALIAS)
#     img.save('FalseCup\\OhneResize\\ohne%d.jpg' % count)
#     count += 1
