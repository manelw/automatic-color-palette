import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from FTC_segmentation import FTC_segmentation  
from rbg_to_hsi_conversion import hsi
from rbg_to_hsi_conversion import rgb 

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

def acopa(image):
    im = cv.imread(image)
    assert im is not None, "file could not be read, check with os.path.exists()"
    grey_cylinder = []
    chromatic_cylinder = []
    Q = 36  # quantification des composants de teinte
    t = Q / (2 * np.pi)  # seuil de saturation pour le cylindre gris
    hsi_image = hsi(image)

    hue = hsi_image[:,0]
    saturation = hsi_image[:,1]
    intensity = hsi_image[:,2]

    # Déterminer le cylindre gris
    for i in range(hsi_image.shape[0]):
            pixel_hsi = hsi_image[i]
            if saturation[i] < t:
                grey_cylinder.append(pixel_hsi)
            else:
                chromatic_cylinder.append(pixel_hsi)

    # Application de la segmentation FTC sur l'histogramme de teinte
  
    S = FTC_segmentation(histogramme(hue))

    # Associer chaque pixel du cylindre gris à son intervalle de teinte
    for i in range(len(grey_cylinder)):
        hue_value = grey_cylinder[i][0]  # extraire la teinte
        for j in range(len(S) - 1):  # `S` contient les points de segmentation
            if S[j] <= hue_value < S[j + 1]:
                grey_cylinder[i] = (j, grey_cylinder[i][1], grey_cylinder[i][2])  # remplacer la teinte par l'indice de segment
                break

    # Création des histogrammes de saturation pour chaque segment de teinte
    for i in range(len(S) - 1):
        Si = []
        for k in range(len(hsi_image)):
            if S[i] <= hue[k] < S[i + 1]:
                Si.append(saturation[k])
        for grey_pixel in grey_cylinder:
            if grey_pixel[0] == i:
                Si.append(grey_pixel[1])

        Si = FTC_segmentation(histogramme(Si))

    # Segmentation de l'intensité pour chaque segment de saturation et de teinte
    for i in range(len(S) - 1):
        for j in range(len(Si) - 1):
            Sij = []
            for k in range(len(hsi_image)):
                    if S[i] <= hue[k] < S[i + 1] and Si[j] <= saturation[k] < Si[j + 1]:
                        Sij.append(intensity[k])
            for grey_pixel in grey_cylinder:
                if grey_pixel[0] == i and Si[j] <= grey_pixel[1] < Si[j + 1]:
                    Sij.append(grey_pixel[2])

            Sij = FTC_segmentation(histogramme(Sij))

    # Calcul des couleurs moyennes pour chaque segment (teinte, saturation, intensité)
    ini_colors = []
    for i in range(len(S) - 1):
        for j in range(len(Si) - 1):
            for k in range(len(Sij) - 1):
                Sijk = []
                for l in range(len(hsi_image)):
                        if (S[i] <= hue[l] < S[i + 1] and
                                Si[j] <= saturation[l] < Si[j + 1] and
                                Sij[k] <= intensity[l] < Sij[k + 1]):
                            Sijk.append((hue[l], saturation[l], intensity[l]))
                for grey_pixel in grey_cylinder:
                    if (grey_pixel[0] == i and
                            Si[j] <= grey_pixel[1] < Si[j + 1] and
                            Sij[k] <= grey_pixel[2] < Sij[k + 1]):
                        Sijk.append(grey_pixel)

                # Calcul de la couleur moyenne en HSI
                if Sijk:
                    mean_color = tuple(np.mean(Sijk, axis=0))  # calcul de la moyenne pour chaque canal H, S, I
                    ini_colors.append(mean_color)
        
        #conversion des couleurs en RGB
        ini_colors = np.array(ini_colors)
        ini_colors = rgb(ini_colors)

    return ini_colors



im = r'./images_test/test6.jpeg'
couleurs = acopa(im)
print("couleurs:", couleurs)



# Affichage des couleurs retournées
for color in couleurs:
    plt.imshow([[color]])
    plt.axis('off')
    plt.show()