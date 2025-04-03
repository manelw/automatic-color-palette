import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def hsi(image):
    #charger l'image
    im = cv.imread(image)
    assert im is not None, "file could not be read, check with os.path.exists()"
    # Convertir l'image en une liste de pixels
    pixels = im.reshape(-1, 3)
    pixels = np.float_(pixels)  # Plus besoin de boucles, tout est dans un tableau NumPy
    R, G, B = pixels[:, 0], pixels[:,1], pixels[:,2]
    r,g,b = R/255, G/255, B/255
     # calcule I
    I = (r + g + b) / 3
    #calcule S
    S = np.sqrt( (r - I)**2 + (g - I)**2 + (b - I)**2)
    ## calcule H

    H = np.arccos( ( ( g - I) - ( b - I))/((S+1e-6)* np.sqrt(2))) ## ajout d'une constante pour éviter division par 0
    
    pixels_hsi = np.stack((H, S, I), axis=-1) ## on empile les 3 matrices pour former une seule matrice

    return pixels_hsi

im = r'./images_test/test6.jpeg'
print(im)
couleurs_hsi = hsi(im)
print("couleurs:", couleurs_hsi)

import numpy as np
import numpy as np

def rgb(pixels_hsi):
    H, S, I = pixels_hsi[:, 0], pixels_hsi[:, 1], pixels_hsi[:, 2]
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Case 1: 0 <= H < 2π/3 (0 <= H < 120°)
    mask1 = (H >= 0) & (H < 2 * np.pi / 3)
    B[mask1] = I[mask1] * (1 - S[mask1])
    R[mask1] = I[mask1] * (1 + S[mask1] * np.cos(H[mask1]) / np.cos(np.pi / 3 - H[mask1]))
    G[mask1] = 3 * I[mask1] - (R[mask1] + B[mask1])
    
    # Case 2: 2π/3 <= H < 4π/3 (120° <= H < 240°)
    mask2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H[mask2] = H[mask2] - 2 * np.pi / 3
    R[mask2] = I[mask2] * (1 - S[mask2])
    G[mask2] = I[mask2] * (1 + S[mask2] * np.cos(H[mask2]) / np.cos(np.pi / 3 - H[mask2]))
    B[mask2] = 3 * I[mask2] - (R[mask2] + G[mask2])
    
    # Case 3: 4π/3 <= H < 2π (240° <= H < 360°)
    mask3 = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H[mask3] = H[mask3] - 4 * np.pi / 3
    G[mask3] = I[mask3] * (1 - S[mask3])
    B[mask3] = I[mask3] * (1 + S[mask3] * np.cos(H[mask3]) / np.cos(np.pi / 3 - H[mask3]))
    R[mask3] = 3 * I[mask3] - (G[mask3] + B[mask3])
    
    # Scale to [0, 255]
    R = np.clip(R * 255, 0, 255)
    G = np.clip(G * 255, 0, 255)
    B = np.clip(B * 255, 0, 255)
    
    pixels_rgb = np.stack((R, G, B), axis=-1).astype(np.uint8)
    return pixels_rgb



"""
## test

im = r'./images_test/fond-vert.jpg'
im1 = cv.imread(im)
pixels = im1.reshape(-1, 3)
print("ori",pixels[0]) 

##obtenir les couleurs en rgb

couleurs_hsi = hsi(im)
print("couleurs hsi", couleurs_hsi[0])
couleurs_rgb = rgb(couleurs_hsi)
print("result rgb",couleurs_rgb[0])
plt.imshow([[couleurs_rgb[0]]])
plt.axis('off')
plt.show()
"""
"""
##v1 pb: on a en entrée les valeur hsi déjà en radiant
def rgb(pixels_hsi):
    H, S, I = pixels_hsi[:, 0], pixels_hsi[:, 1], pixels_hsi[:, 2]
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Convert degrees to radians for trigonometric functions
    H_rad = np.deg2rad(H)
    
    # Case 1: 0 <= H < 120
    mask1 = (H >= 0) & (H < 120)
    B[mask1] = I[mask1] * (1 - S[mask1])
    R[mask1] = I[mask1] * (1 + S[mask1] * np.cos(H_rad[mask1]) / np.cos(np.deg2rad(60) - H_rad[mask1]))
    G[mask1] = 3 * I[mask1] - (R[mask1] + B[mask1])
    
    # Case 2: 120 <= H < 240
    mask2 = (H >= 120) & (H < 240)
    H_rad[mask2] = H_rad[mask2] - np.deg2rad(120)
    R[mask2] = I[mask2] * (1 - S[mask2])
    G[mask2] = I[mask2] * (1 + S[mask2] * np.cos(H_rad[mask2]) / np.cos(np.deg2rad(60) - H_rad[mask2]))
    B[mask2] = 3 * I[mask2] - (R[mask2] + G[mask2])
    
    # Case 3: 240 <= H < 360
    mask3 = (H >= 240) & (H < 360)
    H_rad[mask3] = H_rad[mask3] - np.deg2rad(240)
    G[mask3] = I[mask3] * (1 - S[mask3])
    B[mask3] = I[mask3] * (1 + S[mask3] * np.cos(H_rad[mask3]) / np.cos(np.deg2rad(60) - H_rad[mask3]))
    R[mask3] = 3 * I[mask3] - (G[mask3] + B[mask3])
    
    # Scale to [0, 255]
    R = np.clip(R * 255, 0, 255)
    G = np.clip(G * 255, 0, 255)
    B = np.clip(B * 255, 0, 255)
    
    pixels_rgb = np.stack((R, G, B), axis=-1).astype(np.uint8)
    return pixels_rgb


"""
"""
def rgb(pixels_hsi) :
    H, S, I = pixels_hsi[:,0], pixels_hsi[:,1], pixels_hsi[:,2]
    if ( H >0 & H <120) :
        B = I * (1 - S)
        R = I* [1 + S*np.cos(H)/np.cos(60-H)]
        G = 3*I - (R + B)
    elif (H >= 120 & H < 240) :
        H = H - 120
        R = I * (1 - S)
        G = I* [1 + S*np.cos(H)/np.cos(60-H)]
        B = 3*I - (R + G)
    else : 
        H = H - 240
        G = I * (1 - S)
        B = I* [1 + S*np.cos(H)/np.cos(60-H)]
        R = 3*I - (G + B)
    R = R*255
    G = G*255
    B = B*255
    pixels_rgb = np.stack((R, G, B), axis=-1)
    return pixels_rgb

couleurs_rgb = rgb(couleurs_hsi)
print("couleurs rgb", couleurs_rgb)
"""