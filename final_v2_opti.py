import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from FTC_segmentation import FTC_segmentation  
from rgb_hsi_v4 import hsi  
from rgb_hsi_v4 import rgb
from FTC_segmentation import plot_histogram_and_modes

## méthode 1 pour calculer les histogrammes

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

## méthode 2 pour calculer les histogrammes

def hist1(l, bins=180):
    hist, bin_edges = np.histogram(l, bins=bins)
    return hist, bin_edges

def acopa(image):
    # Chargement de l'image et conversion en HSI
    im = cv.imread(image)
    assert im is not None, "file could not be read, check with os.path.exists()"
    Q = 36
    t = Q / (2 * np.pi)
    hsi_image = hsi(image)
    hue, saturation, intensity = hsi_image[:,0], hsi_image[:,1], hsi_image[:,2]

    # Détermination du cylindre gris et chromatique
    grey_mask = saturation < t
    grey_cylinder = hsi_image[grey_mask]
    chromatic_cylinder = hsi_image[~grey_mask]
    # Segmentation des teintes
    print(hue)
    hist_teintes = hist1(hue)[0]
    plt.plot(hist_teintes)
    S = FTC_segmentation(hist_teintes)
    ##création de la segmentation en valeur de hue
    S = np.array(hist1(hue)[1][S])
    plt.show()
    print("segmentation des teintes", S)
    print(len(hist_teintes))

    # Associer chaque pixel gris à son intervalle de teinte
    grey_indices = np.digitize(grey_cylinder[:, 0], S) - 1
    grey_cylinder[:, 0] = grey_indices

    # Création des histogrammes de saturation pour chaque segment de teinte
    segment_saturations = {i: saturation[(hue >= S[i]) & (hue < S[i + 1])] for i in range(len(S) - 1)}
    print("segment_saturations", segment_saturations)
    for i, seg_saturation in segment_saturations.items():
        grey_sats = grey_cylinder[grey_cylinder[:, 0] == i][:, 1]
        segment_saturations[i] = np.concatenate((seg_saturation, grey_sats))
        print("segment_saturations[i]", segment_saturations[i])
        hist_sat = hist1(segment_saturations[i])
        segment_saturations[i] = FTC_segmentation(hist_sat[0])

        ## conversion de la segmentation en valeur de saturation
        segment_saturations[i] = np.array(hist_sat[1][segment_saturations[i]])

    # Segmentation de l'intensité pour chaque segment de saturation et de teinte
    ini_colors = []
    for i in range(len(S) - 1):
        for j in range(len(segment_saturations[i]) - 1):
            saturation_mask = (saturation >= segment_saturations[i][j]) & (saturation < segment_saturations[i][j + 1])
            hue_mask = (hue >= S[i]) & (hue < S[i + 1])
            Sij_mask = hue_mask & saturation_mask
            
            # Récupérer les pixels d'intensité correspondant à ce segment de teinte et saturation
            Sij_pixels = intensity[Sij_mask]

            # Ajout des pixels gris
            grey_intensity = grey_cylinder[(grey_cylinder[:, 0] == i) &
                                           (grey_cylinder[:, 1] >= segment_saturations[i][j]) &
                                           (grey_cylinder[:, 1] < segment_saturations[i][j + 1])][:, 2]
            Sij_pixels = np.concatenate((Sij_pixels, grey_intensity))
            if Sij_pixels.size ==0: break

            hist = hist1(Sij_pixels)

            Sij_intensity = FTC_segmentation(hist[0])

            ## conversion de la segmentation en valeur d'intensité
            Sij_intensity = np.array(hist[1][Sij_intensity])


            # Calcul des couleurs moyennes pour chaque segment HSI basé sur la segmentation d'intensité
            for k in range(len(Sij_intensity) - 1):
                intensity_mask = (intensity >= Sij_intensity[k]) & (intensity < Sij_intensity[k + 1]) & Sij_mask
                Sijk_pixels = hsi_image[intensity_mask]

                # Ajout des pixels gris
                grey_pixels = grey_cylinder[(grey_cylinder[:, 0] == i) &
                                            (grey_cylinder[:, 1] >= segment_saturations[i][j]) &
                                            (grey_cylinder[:, 1] < segment_saturations[i][j + 1]) &
                                            (grey_cylinder[:, 2] >= Sij_intensity[k]) &
                                            (grey_cylinder[:, 2] < Sij_intensity[k + 1])]
                Sijk_pixels = np.vstack((Sijk_pixels, grey_pixels))

                if len(Sijk_pixels) > 0:
                    mean_color = np.mean(Sijk_pixels, axis=0)
                    ini_colors.append(mean_color)
    
    print(len(ini_colors))
    ini_colors_hsi = np.array(ini_colors)

    #conversion des couleurs en RGB
    ini_colors_rgb = rgb(ini_colors_hsi)
    
    return ini_colors_rgb

"""
#test
im = r'./images_test/test5.jpg'
couleurs = acopa(im)
print("couleurs hsi", couleurs)
couleurs = np.array(couleurs)
couleurs = rgb(couleurs)
print("couleurs:", couleurs)
"""
"""
# Affichage des couleurs retournées
for color in couleurs:
    plt.imshow([[color]])
    plt.axis('off')
    plt.show()
"""
"""
def display_color_grid(colors, grid_size=None):
    """
"""
    Affiche une grille de carrés représentant des couleurs.

    Parameters:
    - colors: Liste ou tableau NumPy contenant les couleurs RGB (valeurs entre 0 et 1 ou 0 et 255).
    - grid_size: Tuple (rows, cols) indiquant la taille de la grille (facultatif).
                 Si None, une grille carrée sera générée automatiquement.
"""
"""
    # Assure-toi que les couleurs sont sous forme de tableau NumPy
    colors = np.array(colors)
    
    # Normalise les couleurs si elles sont entre [0, 255]
 
    
    # Détermine la taille de la grille
    num_colors = len(colors)
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(num_colors)))  # Grille carrée
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_size
    
    # Crée une image vide
    color_grid = np.ones((rows, cols, 3), dtype=np.float32)  # Fond blanc

    # Remplit la grille avec les couleurs
    for idx, color in enumerate(colors):
        row = idx // cols
        col = idx % cols
        if row < rows and col < cols:
            color_grid[row, col] = color

    # Affiche la grille
    plt.figure(figsize=(cols, rows))
    plt.imshow(color_grid)
    plt.axis('off')
    plt.show()

display_color_grid(couleurs)
"""