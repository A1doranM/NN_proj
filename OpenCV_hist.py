import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Train/Yellow/yellow1462.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hue_hist = cv2.calcHist([hsv],[0],None,[180],[0,180])
sat_hist = cv2.calcHist([hsv],[1],None,[256],[0,256])
val_hist = cv2.calcHist([hsv],[2],None,[256],[0,256])

max_hue = np.argmax(hue_hist)
max_sat = np.argmax(sat_hist)
max_val = np.argmax(val_hist)

print("Maximum Hue value:", max_hue, "Maximum Saturation:", max_sat, "Maximum intensity:", max_val)

plt.subplots(figsize=(18, 4))
plt.subplot(131), plt.hist(hsv[:,:,0].ravel(),180,[0,180]), plt.title("hue")
plt.subplot(132), plt.hist(hsv[:,:,1].ravel(),256,[0,256]), plt.title("sat")
plt.subplot(133), plt.hist(hsv[:,:,2].ravel(),256,[0,256]), plt.title("val")
plt.show()