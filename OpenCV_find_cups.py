import cv2
import numpy as np
from matplotlib import pyplot as plt

# #ver 1.
# #cups = cv2.imread("Train/Yellow/yellow1462.jpg")
# cups = cv2.imread("resistor_sm.jpg")
#
# hsv = cv2.cvtColor(cups, cv2.COLOR_BGR2HSV)
#
# lowerbound_b = (5, 75, 150)
# upperbound_b = (30, 125, 200)
#
# # lowerbound_y = (10, 190, 0)
# # upperbound_y = (30, 240, 255)
#
# mask_b = cv2.inRange(hsv, lowerbound_b, upperbound_b)
#
# kernel = np.ones((3,3), np.uint8)
# clean_mask = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)
# clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
# mask_copy_b = clean_mask
# mask_copy_b,contours_b,hierarchy_b = cv2.findContours(mask_copy_b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# image = np.copy(cups)
#
# # mask_y = cv2.inRange(hsv, lowerbound_y, upperbound_y)
# #
# # kernel = np.ones((3,3), np.uint8)
# # clean_mask = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
# # clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
# # mask_copy_y = clean_mask
# # mask_copy_y,contours_y,hierarchy_y = cv2.findContours(mask_copy_y,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # image = np.copy(cups)
# #
# # for i in range(len(contours_y)):
# #     if cv2.contourArea(contours_y[i]) > 2000:
# #         cv2.drawContours(image, contours_y, i, (255, 255, 0), 3)
# #         x, y, w, h = cv2.boundingRect(contours_y[i])
# #         crop = image[y:y + h, x:x + w]
# #
# #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
# #     else:
# #         continue
#
# for i in range(len(contours_b)):
#     if cv2.contourArea(contours_b[i]) > 2000:
#         cv2.drawContours(image, contours_b, i, (255, 255, 0), 3)
#         # x, y, w, h = cv2.boundingRect(contours_b[i])
#         # crop = image[y:y + h, x:x + w]
#         #
#         # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     else:
#         continue
#
# plt.subplots(figsize=(12, 10))
# plt.imshow(image)
# plt.show()

# ver 2.
# cups = cv2.imread("Train/Yellow/yellow1462.jpg")
# cups = cv2.imread("Train/Yellow/yellow1617.jpg")
#
# hsv = cv2.cvtColor(cups, cv2.COLOR_BGR2HSV)
#
# lowerbound_b = (65, 100, 50)
# upperbound_b = (180, 230, 190)
#
# lowerbound_y = (10, 190, 0)
# upperbound_y = (30, 240, 255)
#
# mask_b = cv2.inRange(hsv, lowerbound_b, upperbound_b)
#
# kernel = np.ones((3, 3), np.uint8)
# clean_mask = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)
# clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# mask_copy_b = clean_mask
# mask_copy_b, contours_b, hierarchy_b = cv2.findContours(mask_copy_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# image_b = np.copy(cups)
#
# for i in range(len(contours_b)):
#     if cv2.contourArea(contours_b[i]) > 2000:
#         cv2.drawContours(image_b, contours_b, i, (255, 255, 0), 3)
#         x, y, w, h = cv2.boundingRect(contours_b[i])
#         crop = image_b[y:y + h, x:x + w]
#
#         cv2.rectangle(image_b, (x, y), (x + w, y + h), (0, 0, 255), 3)
#     else:
#         continue
#
# mask_y = cv2.inRange(hsv, lowerbound_y, upperbound_y)
#
# kernel = np.ones((3, 3), np.uint8)
# clean_mask = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
# clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# mask_copy_y = clean_mask
# mask_copy_y, contours_y, hierarchy_y = cv2.findContours(mask_copy_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# image_y = np.copy(cups)
#
# for i in range(len(contours_y)):
#     if cv2.contourArea(contours_y[i]) > 2000:
#         cv2.drawContours(image_y, contours_y, i, (255, 255, 0), 3)
#         x, y, w, h = cv2.boundingRect(contours_y[i])
#         crop = image_y[y:y + h, x:x + w]
#
#         cv2.rectangle(image_y, (x, y), (x + w, y + h), (0, 0, 255), 3)
#     else:
#         continue
#
# plt.subplots(figsize=(12, 10))
# plt.imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
# plt.show()
#
# plt.subplots(figsize=(12, 10))
# plt.imshow(cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB))
# plt.show()


#ONE_CUP

# cups = cv2.VideoCapture("Original videos/yellow_blue.mp4")
#
# while (True):
#     ret, frame = cups.read()
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     lowerbound = (65, 100, 50)
#     upperbound = (180, 230, 190)
#
#     mask = cv2.inRange(hsv, lowerbound, upperbound)
#
#     kernel = np.ones((3, 3), np.uint8)
#
#     clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#
#     mask_copy = clean_mask
#
#     mask_copy, contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     image = np.copy(frame)
#
#     for i in range(len(contours)):
#         if cv2.contourArea(contours[i]) > 2000:
#             cv2.drawContours(image, contours, i, (255, 255, 0), 3)
#             x, y, w, h = cv2.boundingRect(contours[i])
#             crop = image[y:y + h, x:x + w]
#
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
#         else:
#             continue
#
#     plt.subplots(figsize=(12, 10))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.show()
#
# cups.release()
# cv2.waitKey()

#TWO_CUPS_ver.1.

# cups = cv2.VideoCapture("Original videos/VideTraining.MOV")
#
# while (True):
#     ret, frame = cups.read()
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     lowerbound_b = (65, 100, 50)
#     upperbound_b = (180, 230, 190)
#
#     lowerbound_y = (10, 190, 0)
#     upperbound_y = (30, 240, 255)
#
#     mask_b = cv2.inRange(hsv, lowerbound_b, upperbound_b)
#
#     kernel = np.ones((3, 3), np.uint8)
#     clean_mask = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)
#     clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#     mask_copy_b = clean_mask
#     mask_copy_b, contours_b, hierarchy_b = cv2.findContours(mask_copy_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     image_b = np.copy(frame)
#
#     for i in range(len(contours_b)):
#         if cv2.contourArea(contours_b[i]) > 2000:
#             cv2.drawContours(image_b, contours_b, i, (255, 255, 0), 3)
#             x, y, w, h = cv2.boundingRect(contours_b[i])
#             crop = image_b[y:y + h, x:x + w]
#
#             cv2.rectangle(image_b, (x, y), (x + w, y + h), (0, 0, 255), 3)
#         else:
#             continue
#
#     mask_y = cvinRange(hsv, lowerbound_y, upperbound_y)
#
#     kernel = np.ones((3, 3), np.uint8)
#     clean_mask = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
#     clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#     mask_copy_y = clean_mask
#     mask_copy_y, contours_y, hierarchy_y = cv2.findContours(mask_copy_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     image_y = np.copy(frame)
#
#     for i in range(len(contours_y)):
#         if cv2.contourArea(contours_y[i]) > 2000:
#             cv2.drawContours(image_y, contours_y, i, (255, 255, 0), 3)
#             x, y, w, h = cv2.boundingRect(contours_y[i])
#             crop = image_y[y:y + h, x:x + w]
#
#             cv2.rectangle(image_y, (x, y), (x + w, y + h), (0, 0, 255), 3)
#         else:
#             continue
#
#     plt.subplots(figsize=(12, 10))
#     plt.imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
#     plt.show()
#
#     plt.subplots(figsize=(12, 10))
#     plt.imshow(cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB))
#     plt.show()
#
# cups.release()
# cv2.waitKey()

#TWO_CUPS_ver.2.

# cups = cv2.VideoCapture("Original videos/blue_yellow.mp4")
#
# while (True):
#     ret, frame = cups.read()
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     lowerbound_b = (65, 100, 50)
#     upperbound_b = (180, 230, 190)
#
#     lowerbound_y = (10, 190, 0)
#     upperbound_y = (30, 240, 255)
#
#     mask_b = cv2.inRange(hsv, lowerbound_b, upperbound_b)
#
#     kernel = np.ones((3, 3), np.uint8)
#     clean_mask = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)
#     clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#     mask_copy_b = clean_mask
#     mask_copy_b, contours_b, hierarchy_b = cv2.findContours(mask_copy_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     image = np.copy(frame)
#
#     mask_y = cv2.inRange(hsv, lowerbound_y, upperbound_y)
#
#     kernel = np.ones((3, 3), np.uint8)
#     clean_mask = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
#     clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#     mask_copy_y = clean_mask
#     mask_copy_y, contours_y, hierarchy_y = cv2.findContours(mask_copy_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     image = np.copy(frame)
#
#     for i in range(len(contours_y)):
#         if cv2.contourArea(contours_y[i]) > 2000:
#             cv2.drawContours(image, contours_y, i, (255, 255, 0), 3)
#             # x, y, w, h = cv2.boundingRect(contours_y[i])
#             # crop = image[y:y + h, x:x + w]
#             #
#             # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         else:
#             continue
#
#     for i in range(len(contours_b)):
#         if cv2.contourArea(contours_b[i]) > 2000:
#             cv2.drawContours(image, contours_b, i, (255, 255, 0), 3)
#             # x, y, w, h = cv2.boundingRect(contours_b[i])
#             # crop = image[y:y + h, x:x + w]
#             #
#             # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
#         else:
#             continue
#
#     plt.subplots(figsize=(12, 10))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.show()
#
# cups.release()
# cv2.waitKey()

#######################################################################################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt

cup = cv2.imread("Train/Blue/blue719.jpg")
hsv = cv2.cvtColor(cup, cv2.COLOR_BGR2HSV)
lowerbound = (20, 210, 0)
upperbound = (32, 250, 255)
mask1 = cv2.inRange(hsv, lowerbound, upperbound)

# /////////////////////////////////////////////////////////////////////
kernel = np.ones((3,3), np.uint8)
img = cv2.imread("Train/Blue/blue719.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cup = cv2.imread("Train/Blue/blue719.jpg")
hsv = cv2.cvtColor(cup, cv2.COLOR_BGR2HSV)
lowerbound = (25, 235, 0)
upperbound = (50, 255, 255)
mask1 = cv2.inRange(hsv, lowerbound, upperbound)

# use opening to remove the white specks (noise)
clean_mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
# use closing to fill up tears. Bettwe use a bigger kernel size
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
# ////////////////////////////////////////////////////////////////
mask_copy = clean_mask

mask_copy,contours,hierarchy = cv2.findContours(mask_copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
image = np.copy(cup)

for i in range(len(contours)):
    cv2.drawContours(image, contours, i, (255,255,255), 3)

image2 = np.copy(cup)
for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 5000:
            x, y, w, h = cv2.boundingRect(contours[i])
            # crop = image2[y:y + h, x:x + w]
            # cv2.imwrite("crop_" + str(i) + ".jpg", crop)

            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        else:
            continue
image3 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
image4 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)

# ////////////////////////////////////////////////////////////////////
plt.subplots(figsize=(14, 10))
plt.subplot(121), plt.imshow(mask1, cmap='gray')
plt.subplot(122), plt.imshow(clean_mask, cmap='gray')


plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
# plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
#


hue_hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
sat_hist = cv2.calcHist([hsv],[1],None,[256],[0,256])
val_hist = cv2.calcHist([hsv],[2],None,[256],[0,256])

max_hue = np.argmax(hue_hist)
max_sat = np.argmax(sat_hist)
max_val = np.argmax(val_hist)

# //////////////////////////////////////////////////////////////////////////




print ("Maximum Hue value:", max_hue, "Maximum Saturation:", max_sat, "Maximum intensity:", max_val)

# plt.subplots(figsize=(18, 4))
# plt.subplot(131), plt.hist(hsv[:,:,0].ravel(),180,[0,180]), plt.title("hue")
# plt.subplot(132), plt.hist(hsv[:,:,1].ravel(),256,[0,256]), plt.title("sat")
# plt.subplot(133), plt.hist(hsv[:,:,2].ravel(),256,[0,256]), plt.title("val")
# plt.show()

cv2.waitKey()
cv2.destroyAllWindows()