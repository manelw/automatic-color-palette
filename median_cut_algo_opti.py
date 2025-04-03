import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def median_cut(image, n_colors):
    # Charger l'image
    im = cv.imread(image)
    assert im is not None, "file could not be read, check with os.path.exists()"

    # Convertir l'image en une liste de pixels
    pixels = im.reshape(-1, 3)  # Plus besoin de boucles, tout est dans un tableau NumPy

    # Initialiser la liste des buckets avec tous les pixels
    buckets = [pixels]

    # Boucle tant que le nombre de buckets est inférieur au nombre de couleurs souhaité
    while len(buckets) < n_colors:
        # Trouver le bucket à diviser
        new_buckets = []
        for bucket in buckets:
            if len(bucket) == 0:
                continue

            # Calculer la plage de chaque canal (R, G, B)
            r_range = np.ptp(bucket[:, 0])  # np.ptp donne la range (max - min)
            g_range = np.ptp(bucket[:, 1])
            b_range = np.ptp(bucket[:, 2])

            # Choisir le canal avec la plus grande plage
            channel = np.argmax([r_range, g_range, b_range])

            # Trier les pixels par la valeur du canal choisi
            bucket = bucket[bucket[:, channel].argsort()]

            # Séparer le bucket en deux
            half = len(bucket) // 2
            new_buckets.append(bucket[:half])
            new_buckets.append(bucket[half:])

        buckets = new_buckets

    # Calculer la couleur moyenne de chaque bucket
    palette = [np.mean(bucket, axis=0) for bucket in buckets if len(bucket) > 0]

    return palette

# Test
image = r'./images_test/test1.jpg'
n_colors = 9
palette = median_cut(image, n_colors)

# Fonction pour afficher les couleurs
def display_palette(palette):
    fig, ax = plt.subplots(1, len(palette), figsize=(len(palette) * 2, 2))

    for i, color in enumerate(palette):
        ax[i].imshow([[color / 255]])  # Diviser par 255 pour normaliser les couleurs (0-1)
        ax[i].axis('off')  # Enlever les axes pour un meilleur affichage

    plt.show()

display_palette(palette)

for color in palette:
    print(color)

## attribuer aux pixels de l'image les couleurs de la palette

def assign_colors(image, palette):
    #charger l'image
    im = cv.imread(image)
    assert im is not None, "file could not be read, check with os.path.exits()"
    #convertir l'image en une liste de pixels
    pixels = im.reshape(-1,3)
    #initialiser la liste des couleurs
    colors = []
    #boucle sur les pixels
    for pixel in pixels:
        #calculer la distance entre le pixel et chaque couleur de la palette.
        distances = np.linalg.norm(palette - pixel, axis=1)
        #trouver l'indice de la couleur la plus proche
        closest_index = np.argmin(distances)
        #attribuer la couleur correspondante
        colors.append(palette[closest_index])
        #convertir la liste en tableau NumPy
    colors = np.array(colors, dtype=np.uint8)
        #convertir le tableau en image
    output = colors.reshape(im.shape)
    return output

# Test

output = assign_colors(image,palette)

cv.imshow('output',output)
        
# Attendre qu'une touche soit pressée
cv.waitKey(0)

# Fermer toutes les fenêtres ouvertes
cv.destroyAllWindows()