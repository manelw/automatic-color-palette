import cv2
import numpy as np
import matplotlib.pyplot as plt



def hsi(image):
    img = image  # Change the filename and path according to your need
    rgbimg = cv2.cvtColor(cv2.imread(img,cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # Convert rgb images to hsi images
    # Hint: Normalize Hue value to [0,255] for demonstration purpose.
    rows, cols = rgbimg[:,:,0].shape  # We expect that for each channel image dims are same
    s = np.zeros((rows, cols), dtype=np.float32)  # Initialize s
    i = np.zeros((rows, cols), dtype=np.float32)  # Initialize i
    red = rgbimg[:,:,0]/255  # Normalize to 0-1
    green = rgbimg[:,:,1]/255
    blue = rgbimg[:,:,2]/255
    h = []
    for r in range(rows):
        for c in range(cols):
            RG = red[r,c]-green[r,c]+0.001  # Red-Green, add a constant to prevent undefined value
            RB = red[r,c]-blue[r,c]+0.001  # Red-Blue
            GB = green[r,c]-blue[r,c]+0.001  # Green-Blue
            theta = np.arccos(np.clip(((0.5*(RG+RB))/(RG**2+RB*GB)**0.5), -1, 1))  # Still in radians
            theta = np.degrees(theta)  # Convert to degrees
            if blue[r,c] <= green[r,c]:
                h.append(theta)
            else:
                h.append(360 - theta)
    # Hue range will be automatically scaled to 0-255 by matplotlib for display
    # We will need to convert manually to range of 0-360 in hsi2rgb function
    h = np.array(h, dtype=np.int64).reshape(rows, cols)  # Convert Hue to NumPy array
    h = ((h - h.min()) * (1/(h.max() - h.min()) * 360))  # Scale h to 0-360
    minRGB = np.minimum(np.minimum(red, green), blue)
    s = 1-((3/(red+green+blue+0.001))*minRGB)  # Add 0.001 to prevent divide by zero
    i = (red+green+blue)/3  # Intensity: 0-1
    ## on applatie les tableaux car pas besoin des valeurs des lignes/colonnes pour notre usage
    h = h.reshape(-1)
    s = s.reshape(-1)
    i = i.reshape(-1)
    pixels_hsi = np.stack((h, s, i), axis=-1) ## on empile les 3 matrices pour former une seule matrice
    return pixels_hsi

def rgb(hsiimg):
    # Convert hsi images to rgb images
    #rows, cols = hsiimg[:,:,0].shape  # We expect that for each channel image dims are same
    h = hsiimg[:,0]  # 0-360
    l = len(h)
    h = ((h - h.min()) * (1/(h.max() - h.min()) * 360))  # Scale h to 0-360
    s = hsiimg[:,1]  # 0-1
    i = hsiimg[:,2]  # 0-1
    rd, gr, bl = [], [], []  # Initialize r, g, and b as empty array
    for k in range(l):
            if (h[k] >= 0 and h[k] <= 120):
                red = (1+((s[k]*np.cos(np.radians(h[k])))/np.cos(np.radians(60-h[k]))))/3
                blue = (1-s[k])/3
                rd.append(red)
                gr.append(1-(red+blue))
                bl.append(blue)
            elif (h[k] > 120 and h[k] <= 240):
                h[k] = h[k]-120
                red = (1-s[k])/3
                green = (1+((s[k]*np.cos(np.radians(h[k])))/np.cos(np.radians(60-h[k]))))/3
                rd.append(red)
                gr.append(green)
                bl.append(1-(red+green))
            elif (h[k] > 240 and h[k] <= 360):
                h[k] = h[k]-240
                green = (1-s[k])/3
                blue = (1+((s[k]*np.cos(np.radians(h[k])))/np.cos(np.radians(60-h[k]))))/3
                rd.append(1-(green+blue))
                gr.append(green)
                bl.append(blue)
    rd = np.multiply(rd, 3*i.flatten())# R = r*3*i, where r = rd in previous row
    gr = np.multiply(gr, 3*i.flatten())
    bl = np.multiply(bl, 3*i.flatten())
    ## on applatie les tableaux car pas besoins des valeurs des coordonnées pour notre usage
    rd = rd.reshape(-1)
    gr = gr.reshape(-1)
    bl = bl.reshape(-1)
    pixels_rgb = np.stack((rd, gr, bl), axis=-1)
    return pixels_rgb