import numpy as np
import cv2 as cv

def hsi(image):
    # Charger l'image
    im = cv.imread(image)
    assert im is not None, "Le fichier n'a pas pu être lu, vérifie avec os.path.exists()"
    
    # Conserver les dimensions d'origine
    height, width, _ = im.shape
    
    # Convertir l'image en une liste de pixels
    pixels = im.reshape(-1, 3)
    pixels = np.float32(pixels)  # Conversion en float
    
    # Normalisation des valeurs RGB entre [0, 1]
    R, G, B = pixels[:, 0] / 255, pixels[:, 1] / 255, pixels[:, 2] / 255
    
    # Calcul de l'intensité (I)
    I = (R + G + B) / 3.0
    
    # Calcul de la saturation (S)
    min_rgb = np.minimum(np.minimum(R, G), B)
    delta = (R + G + B) - 3 * min_rgb
    S = np.zeros_like(delta)
    S[I > 0] = 1 - min_rgb[I > 0] / I[I > 0]
    
    # Calcul de la teinte (H)
    H = np.zeros_like(R)
    
    # Masques pour les pixels où delta > 0
    mask_non_gray = delta > 0
    
    # Calcul de la teinte pour les pixels non-gris
    R_non_gray, G_non_gray, B_non_gray, delta_non_gray = R[mask_non_gray], G[mask_non_gray], B[mask_non_gray], delta[mask_non_gray]
    
    # Cas où le rouge est dominant
    mask_r = (R_non_gray >= G_non_gray) & (R_non_gray >= B_non_gray)
    H[mask_non_gray][mask_r] = (G_non_gray[mask_r] - B_non_gray[mask_r]) / delta_non_gray[mask_r]
    
    # Cas où le vert est dominant
    mask_g = (G_non_gray >= R_non_gray) & (G_non_gray >= B_non_gray)
    H[mask_non_gray][mask_g] = 2 + (B_non_gray[mask_g] - R_non_gray[mask_g]) / delta_non_gray[mask_g]
    
    # Cas où le bleu est dominant
    mask_b = (B_non_gray >= R_non_gray) & (B_non_gray >= G_non_gray)
    H[mask_non_gray][mask_b] = 4 + (R_non_gray[mask_b] - G_non_gray[mask_b]) / delta_non_gray[mask_b]
    
    # Conversion de H en degrés
    H[mask_non_gray] *= 60
    H[H < 0] += 360  # Ajuster les valeurs négatives
    
    # Rassembler les composants HSI
    pixels_hsi = np.stack((H, S, I), axis=-1)
    return pixels_hsi
    #hsi_image = pixels_hsi.reshape(height, width, 3)  # Restaurer la forme originale
"""
# Exemple d'appel de la fonction
image_path = r'./images_test/fond-vert.jpg'
hsi_image = hsi(image_path)

# Afficher un exemple de la conversion HSI
print(hsi_image.shape)

"""
def rgb(image_hsi):
    # Extraire les trois canaux H, S, I
    h, s, i = image_hsi[:, 0], image_hsi[:,1], image_hsi[:,2]
    
    # Normaliser la teinte à [0, 1]
    h = h / 360.0
    
    # Initialiser les canaux RGB
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    # Traiter les pixels avec saturation nulle (gris) - pas besoin de calculer la teinte
    mask_gray = (s == 0)
    r[mask_gray] = g[mask_gray] = b[mask_gray] = i[mask_gray]

    # Traiter les pixels colorés (saturation non nulle)
    mask_colored = ~mask_gray
    
    # Pour les pixels colorés, procéder selon la teinte (H)
    # Calcul de la conversion pour chaque plage de teinte : 0 - 120, 120 - 240, 240 - 360
    # Calcul des composantes RGB pour chaque teinte

    # Cas où 0 <= H < 120
    mask_0_120 = (h >= 0) & (h < 120)
    b[mask_0_120] = i[mask_0_120] * (1 - s[mask_0_120])
    r[mask_0_120] = i[mask_0_120] * (1 + s[mask_0_120] * np.cos(np.radians(h[mask_0_120])) / 
                                      (np.cos(np.radians(60 - h[mask_0_120])) + 1e-10))
    g[mask_0_120] = 3 * i[mask_0_120] - (r[mask_0_120] + b[mask_0_120])

    # Cas où 120 <= H < 240
    mask_120_240 = (h >= 120) & (h < 240)
    h[mask_120_240] -= 120  # Ajustement pour la plage [0, 120] dans l'espace H
    r[mask_120_240] = i[mask_120_240] * (1 - s[mask_120_240])
    g[mask_120_240] = i[mask_120_240] * (1 + s[mask_120_240] * np.cos(np.radians(h[mask_120_240])) / 
                                          (np.cos(np.radians(60 - h[mask_120_240])) + 1e-10))
    b[mask_120_240] = 3 * i[mask_120_240] - (r[mask_120_240] + g[mask_120_240])

    # Cas où 240 <= H < 360
    mask_240_360 = (h >= 240) & (h < 360)
    h[mask_240_360] -= 240  # Ajustement pour la plage [0, 120] dans l'espace H
    g[mask_240_360] = i[mask_240_360] * (1 - s[mask_240_360])
    b[mask_240_360] = i[mask_240_360] * (1 + s[mask_240_360] * np.cos(np.radians(h[mask_240_360])) / 
                                          (np.cos(np.radians(60 - h[mask_240_360])) + 1e-10))
    r[mask_240_360] = 3 * i[mask_240_360] - (g[mask_240_360] + b[mask_240_360])

    # Empiler les résultats RGB
    rgb_image = np.stack((r, g, b), axis=-1)

    # Convertir les résultats à la plage [0, 255] et cliper les valeurs pour éviter les débordements
    return np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

"""
# Exemple d'appel de la fonction
rbg_image = rbg(hsi_image)
print(rbg_image.shape)  # Affiche la forme du tableau RGB résultant
"""
