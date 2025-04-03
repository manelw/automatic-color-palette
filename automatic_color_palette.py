from  FTC_segmentation import FTC_segmentation
import numpy as np
from rbg_to_hsi_conversion import hsi,rgb
from matplotlib import pyplot as plt
import cv2 as cv

def histogramme_r(list, r, pas):
    a = 0
    for i in range(len(list)):
        #print(list[i])
        if np.abs(list[i]-r)<pas:
            a += 1
    return a

def histogramme(list):
    maxr = max(list)
    return [histogramme_r(list, r, maxr/180) for r in np.linspace(0, maxr, 180)]

def ACoPa(image):
    pixels_hsi = hsi(image)
    hue = np.array(pixels_hsi[:,0])
    saturation = np.array(pixels_hsi[:,1])
    intensity = np.array(pixels_hsi[:,2])
    print("calculating modes_h")
    hist = histogramme(hue)
    modes_h = FTC_segmentation(hist)
    print(modes_h)
    s_list = [[] for _ in range(len(modes_h) - 1)]

    for i in range(len(modes_h)-1):
        s_list[i] = saturation[modes_h[i]:modes_h[i+1]]
    print(s_list)
    modes_s = [[] for _ in range(len(modes_h) - 1)]
    for i in range(len(s_list)):
        modes_s[i] = FTC_segmentation(histogramme(s_list[i]))
    i_list = [[] for _ in range(len(modes_h) - 1)]
    for i in range(len(modes_s)) : 
        i_list[i] = [[] for _ in range(len(modes_h) - 1)]
        for j in range(len(modes_s[i])-1) :
             i_list[i][j] = (intensity[modes_s[i][j]:modes_s[i][j+1]])
    modes_i = [[] for _ in range(len(modes_h) - 1)]
    for i in range(len(i_list)) : 
        modes_i[i] = [[] for _ in range(len(modes_h) - 1)]
        for j in range(len(i_list[i])) :
            modes_i[i][j] = FTC_segmentation(histogramme(i_list[i][j]))
    return(modes_i)

im = r'./images_test/test6.jpeg'
couleurs = ACoPa(im)
print("couleurs:", couleurs)

# Affichage des couleurs retournées
for color in couleurs:
    plt.imshow([[color]])
    plt.axis('off')
    plt.show()

