import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from FTC_segmentation.py import FTC_segmentation
from rbg_to_hsi_conversion.py import hsi

def acopa(image):
    im = cv.imread(image)

    grey_cylinder = []
    chromatic_cylinder = []
    Q = 36 ## quantification des hue components
    t = Q/(2*np.pi) ## treshold, rayon du grey cylinder
    hsi_image = hsi(image)

    hue = hsi_image[:, :, 0]
    saturation = hsi_image[:, :, 1]
    intensity = hsi_image[:, :, 2]

    #détermination du grey cylinder
    for i in range(hsi_image.shape[0]):
        for j in range(hsi_image.shape[1]):
            if saturation[i,j] <t:
                grey_cylinder.append(hsi_image[i,j])
            else: 
                chromatic_cylinder.append(hsi_image[i,j])


    #Apply the FTC algorithm on the hue histogram of the image. Let S be the obtained segmentation.

    S = FTC_segmentation(hsi_image[:,:,0])

    #2. Link each pixel of the grey cylinder to its corresponding interval Si = [si,si+1], according to its hue value.

    for i in range(len(grey_cylinder)):
        for j in range(len(S)):
            if S[j][0] <= grey_cylinder[i][0] <= S[j][1]:
                grey_cylinder[i][0] = j 
                break

    #3 For each i, construct the saturation histogram of all the pixels in the image whose hue belongs to Si. 
    # Take into account the pixels of the grey cylinder. 
    # Apply the FTC algorithm on the corresponding saturation histogram. For each i, let {Si,1,Si,2,...} be the obtained segmentation.
    
    for i in range(len(S)):
        Si = []
        for j in range(hsi_image.shape[0]):
            for k in range(hsi_image.shape[1]):
                if S[i][0] <= hsi_image[j,k,0] <= S[i][1]:
                    Si.append(hsi_image[j,k,1])     # on met les valeurs de saturation des pixels dont la teinte appartient à S[i] dans Si
        for j in range(len(grey_cylinder)):         # take into account pixels of the grey cylinder
            if S[i][0] <= grey_cylinder[j][0] <= S[i][1]:
                Si.append(grey_cylinder[j][1])
        
        Si = np.array(Si)
        Si = FTC_segmentation(Si)

    #4. For each i and each j, compute and segment the intensity histogram of all the
    #pixels whose hue and saturation belong to Si and Si,j, including those in the grey cylinder.

    for i in range(len(S)):
        for j in range(len(Si)):
            Sij = []
            for k in range(hsi_image.shape[0]):
                for l in range(hsi_image.shape[1]):
                    if S[i][0] <= hsi_image[k,l,0] <= S[i][1] and Si[j][0] <= hsi_image[k,l,1] <= Si[j][1]:
                        Sij.append(hsi_image[k,l,2])
            for k in range(len(grey_cylinder)):
                if S[i][0] <= grey_cylinder[k][0] <= S[i][1] and Si[j][0] <= grey_cylinder[k][1] <= Si[j][1]:
                    Sij.append(grey_cylinder[k][2])

            Sij = np.array(Sij)
            Sij = FTC_segmentation(Sij)

    #5. For each i, j and k, compute the mean color value of all the pixels whose hue, saturation and
    #intensity belong to Si, Sij and Sijk including those in the grey cylinder

    ini_colors=[]

    for i in range(len(S)):
        for j in range(len(Si)):
            for k in range(len(Sij)):
                Sijk = []
                for l in range(hsi_image.shape[0]):
                    for m in range(hsi_image.shape[1]):
                        if S[i][0] <= hsi_image[l,m,0] <= S[i][1] and Si[j][0] <= hsi_image[l,m,1] <= Si[j][i] and Sij[k][0] <= hsi_image[l,m,2] <= Sij[k][1]:
                            Sijk.append(hsi_image[l,m])
                for l in range(len(grey_cylinder)):
                    if S[i][0] <= grey_cylinder[l][0] <= S[i][1] and Si[j][0] <= grey_cylinder[l][1] <= Si[j][1] and Sij[k][0] <= grey_cylinder[l][2] <= Sij[k][1]:
                        Sijk.append(grey_cylinder[l])
            #on veut juste les moyennes des 3 valeurs de hsi pas les coordonnées
            mean_color = np.mean(Sijk, axis=0)
            ini_colors.append(mean_color)
    return ini_colors
            
    
    
    ##à optimiser
    ## mean color combien il y a d'arguments dans un élément de mean color? Juste les 3 valeurs de hsi? Ou y a les coordonnées en plus?

    #test avec une image
    im = cv.imread('test1.jpg')
    couleurs = acopa(im)
    print(couleurs)

    #afficher les couleurs retournées par acopa
    for i in range(len(couleurs)):
        plt.imshow(couleurs[i])






            



 