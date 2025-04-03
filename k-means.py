import numpy as np 
from matplotlib import pyplot as plt
import scipy
from PIL import Image


image = plt.imread(r'./images_test/albane.jpg')

clusters = 10

def distance_rgb (pix1, pix2):
    return np.sqrt((pix1[0] - pix2[0])**2 + (pix1[1] - pix2[1])**2 + (pix1[2] - pix2[2])**2)

def init_centroides(clusters,image) : 
    indices_x = np.random.randint(0, image.shape[0], size=clusters)
    indices_y = np.random.randint(0, image.shape[1], size=clusters)
    centroides = image[indices_x, indices_y]
    return centroides
    


def recalcule_centroides(partitions, centroides, image):

    for i in range(clusters):
        points = partitions[i]
        if points:
            coords = np.array(points)
            avg = np.mean(image[coords[:, 0], coords[:, 1]], axis=0)
            centroides[i] = avg
    return centroides

def iter_k_means(image, centroides):
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

def k_means(image,clusters,iterations) : 
    centroides = init_centroides(clusters,image)
    
    for i in range(iterations) : 
        partitions = iter_k_means(image,centroides)
        centroides = recalcule_centroides(partitions,centroides,image)
    return centroides, partitions
        
centroides,partitions = k_means(image,clusters,10)


def apply_partitions(image, partitions, centroides):
    new_image = np.copy(image)
    for i in range (clusters) : 
         cluster = partitions[i]
         for x,y in cluster : 
              new_image[x,y] = centroides[i]     
    return new_image

# Example usage
new_image = apply_partitions(image, partitions, centroides)
img = Image.fromarray(new_image)
img.save('k-means.jpg')
img.show()
