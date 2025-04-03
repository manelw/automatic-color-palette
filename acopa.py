import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from final_v2_opti import acopa

# Charger l'image
#image = plt.imread('./images_test/test6.jpeg')
image_name = r'./images_test/cocci_petit.jpg'


# Paramètres
clusters = 20  # Nombre de clusters souhaités
iterations = 20  # Nombre d'itérations

import numpy as np 
from matplotlib import pyplot as plt
import scipy
from PIL import Image


clusters = 20

def distance_rgb (pix1, pix2):
    return np.sqrt((pix1[0] - pix2[0])**2 + (pix1[1] - pix2[1])**2 + (pix1[2] - pix2[2])**2)

def init_centroides(image) : 
    centroides = acopa(image)
    return centroides
    


def recalcule_centroides(partitions, centroides, image):
    image = plt.imread(image)
    for i in range(len(centroides)):
        points = partitions[i]
        if points:
            coords = np.array(points)
            avg = np.mean(image[coords[:, 0], coords[:, 1]], axis=0)
            centroides[i] = avg
    return centroides

def iter_k_means(image, centroides):
    image = plt.imread(image)
    # Initialiser les partitions
    nv_partitions = [[] for _ in range(len(centroides))]
    
    # Reshape l'image pour avoir une liste de pixels
    pixels = image.reshape(-1, image.shape[2])
    
    # Calculer les distances entre chaque pixel et les centroides
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroides, axis=2)
    
    # Trouver le cluster le plus proche pour chaque pixel
    clusters = np.argmin(distances, axis=1)
    
    # Reshape les indices des pixels pour correspondre à l'image originale
    indices = np.indices(image.shape[:2]).reshape(2, -1).T
    
    # Assigner chaque pixel au cluster correspondant
    for idx, cluster in zip(indices, clusters):
        nv_partitions[cluster].append(tuple(idx))
    
    return nv_partitions

def k_means(image,iterations) : 
    centroides = init_centroides(image)
    
    for i in range(iterations) : 
        partitions = iter_k_means(image,centroides)
        centroides =recalcule_centroides(partitions,centroides,image)
    return centroides, partitions
        
centroides,partitions = k_means(image_name,20)


def apply_partitions(image, partitions, centroides):
    image = plt.imread(image)
    new_image = np.copy(image)
    for i in range (len(centroides)) : 
         cluster = partitions[i]
         for x,y in cluster : 
              new_image[x,y] = centroides[i]     
    return new_image

# Example usage
new_image = apply_partitions(image_name, partitions, centroides)
img = Image.fromarray(new_image)
img.show()


"""

# Fonction de distance RGB
def distance_rgb(pix1, pix2):
    return np.sqrt((pix1[0] - pix2[0])**2 + (pix1[1] - pix2[1])**2 + (pix1[2] - pix2[2])**2)

# Initialisation des centroïdes à partir de `acopa`
def init_centroides_acopa(image):
    # Utiliser acopa pour obtenir les couleurs initiales
    centroides = np.array(acopa(image))

    # Vérifier si les valeurs sont entre 0 et 255 et normaliser si nécessaire
    if centroides.max() > 1:
        centroides = centroides / 255.0

    return centroides
    

# Recalcul des centroïdes
def recalcule_centroides(partitions, centroides, image):
    image=plt.imread(image)
    for i in range(len(centroides)):
        points = partitions[i]
        if points:  # Si le cluster n'est pas vide
            coords = np.array(points)
            avg = np.mean(image[coords[:, 0], coords[:, 1]], axis=0)
            centroides[i] = avg
    return centroides

# Mise à jour des partitions
def iter_k_means(image, centroides):
    image = plt.imread(image)
    nv_partitions = [[] for _ in range(len(centroides))]
    pixels = image.reshape(-1, image.shape[2])
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroides, axis=2)
    clusters = np.argmin(distances, axis=1)
    indices = np.indices(image.shape[:2]).reshape(2, -1).T
    for idx, cluster in zip(indices, clusters):
        nv_partitions[cluster].append(tuple(idx))
    return nv_partitions

# Algorithme de K-Means
def k_means(image, clusters, iterations):
    centroides = init_centroides_acopa(image)  # Initialisation avec acopa
    for i in range(iterations):
        partitions = iter_k_means(image, centroides)
        centroides = recalcule_centroides(partitions, centroides, image)
    return centroides, partitions

# Application des partitions pour créer une image segmentée
def apply_partitions(image, partitions, centroides):
    image = plt.imread(image)
    new_image = np.copy(image)
    for i in range(len(centroides)):
        cluster = partitions[i]
        for x, y in cluster:
            new_image[x, y] = centroides[i]
    return new_image

# Exécution de K-Means
centroides, partitions = k_means(image_name, clusters, iterations)

# Application des clusters à l'image
new_image = apply_partitions(image_name, partitions, centroides)
img = Image.fromarray((new_image * 255).astype(np.uint8))  # Convertir en uint8 si nécessaire
img.show()

"""