import numpy as np 
import matplotlib
import scipy
import scipy.linalg
import scipy.signal
import scipy.stats
from matplotlib import pyplot as plt
import cv2 as cv


def grenander_down(h) :
    n = len(h)
    gren = np.copy(h)
    for j in range(n) :
        i = 0 
        while i < n-1 and gren[i]>= gren[i+1] :
            i+=1
        if  i == n-1 :
            break
        j = i 
        while j < n-1 and gren[j] <= gren[j+1] :
            j+=1
        mean_val = np.mean(gren[i:j+1])
        gren[i:j+1] = mean_val
        i = j+1

    return gren


def plot_histogram_and_grenander_down(h):
    grenander = grenander_down(h)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the original histogram
    plt.subplot(1, 2, 1)
    plt.plot(h, label='Histogram')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Original Histogram')
    plt.legend()
    
    # Plot the Grenander estimator
    plt.subplot(1, 2, 2)
    plt.plot(grenander, label='Grenander Down', color='orange')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Grenander Down Estimator')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def grenander_up(h) :
    n = len(h)
    gren = np.copy(h)
    for j in range(n) :
        i = 0 
        while i < n-1 and gren[i]<= gren[i+1] :
            i+=1
        if  i == n-1 :
            break
        j = i 
        while j < n-1 and gren[j] >= gren[j+1] :
            j+=1
        mean_val = np.mean(gren[i:j+1])
        gren[i:j+1] = mean_val
        i = j+1

    return gren


def plot_histogram_and_grenander_up(h,file_path=None):
    grenander = grenander_up(h)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the original histogram
    plt.subplot(1, 2, 1)
    plt.plot(h, label='Histogram')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Original Histogram')
    plt.legend()
    
    # Plot the Grenander estimator
    plt.subplot(1, 2, 2)
    plt.plot(grenander, label='Grenander up', color='orange')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Grenander up estimator')
    plt.legend()
    
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    else:
        plt.show()

def generate_decreasing_with_noise(size, start, end, noise_level):
    decreasing = np.linspace(start, end, size)
    noise = np.random.normal(0, noise_level, size)
    return decreasing + noise

#decreasing_with_noise = generate_decreasing_with_noise(100, 10, 0, 0.5)
#plot_histogram_and_grenander_down(decreasing_with_noise)

#increasing_with_noise = generate_decreasing_with_noise(100,0,10,0.5)
#plot_histogram_and_grenander_up(increasing_with_noise, file_path="increasing_with_noise.png")


def hyp_decr(h) :
    gren = grenander_down(h)
    k = scipy.stats.ks_2samp(h,gren)[1]
    return  k >= 0.1

def hyp_incr(h) :
    gren = grenander_up(h)
    k = scipy.stats.ks_2samp(h,gren)[1]
    return  k >= 0.1

def unimodal(h) : 
    n = len(h)
    for i in range(1,n-1) :
        h1 = h[:i]
        h2 = h[i:]
        if (hyp_incr(h1) and hyp_decr(h2)) :
            return True
    return False




def loc_min(h) : 
    l = []
    if h[0] < h[1] : 
        l.append(0)
    for i in range(1,len(h)-1) :
        if (h[i] < h[i-1] and  h[i] <= h[i+1] or h[i] <= h[i-1] and  h[i] < h[i+1]) : 
            l.append(i)
    if h[-1] < h[-2] :
        l.append(len(h)-1)
    return np.array(l)

def merge_segments(h, S):
    modified = False
    i = 1
    while i < len(S) - 1:
        merged_interval = h[S[i-1]:S[i+1] + 1]
        if unimodal(merged_interval):
            S = np.delete(S, i)
            modified = True
            #print("Merged segments at index", i)
        else:
            i += 1
    return S, modified


def FTC_segmentation(h):
    S = loc_min(h)
    #print("Initial segmentation:", S)
    
    # Step 2: Merge segments based on the unimodal hypothesis
    while True:
        S, modified = merge_segments(h, S)
        if not modified:
            break
    
    # Step 3: Repeat with unions of multiple segments
    for group_size in range(3, len(S)):
        while True:
            modified = False
            i = 1
            while i < len(S) - group_size + 1:
                merged_interval = h[S[i-1]:S[i + group_size - 1] + 1]
                if unimodal(merged_interval):
                    S = np.delete(S, range(i, i + group_size - 1))
                    modified = True
                    #print("Merged segments of size", group_size, "at index", i)
                else:
                    i += 1
            if not modified:
                break
    
    
    return S

   
def plot_histogram_and_modes(h, modes,file_path=None):
    print("Modes:", modes)
    plt.figure(figsize=(10, 6))
    plt.plot(h, label='Histogram')
    #plt.scatter(modes, h[modes], color='red', label='Modes')
    # Ajout des barres rouges pour les modes
    bar_width = 1  # Largeur des barres rouges
    for mode in modes:
        print(mode)
        plt.bar(mode, max(h), color='red', width=bar_width)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Histogram and Modes')
    plt.legend()
    if file_path:
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    plt.show()
    
    
"""

def generate_increasing_decreasing_distribution(size, peak):
    return np.concatenate((np.linspace(0, peak, size // 2), np.linspace(peak, 0, size // 2)))

# Generate a small increasing then decreasing distribution
small_distribution = generate_increasing_decreasing_distribution(10, 5)

# Generate a much larger increasing then decreasing distribution
large_distribution = generate_increasing_decreasing_distribution(100, 50)


zeros = np.zeros(10)

distribution = np.concatenate((small_distribution, zeros, large_distribution))
#plot_histogram_and_modes(distribution, FTC_segmentation(distribution))

distribution = np.concatenate((zeros, large_distribution, zeros, large_distribution, zeros))    
#plot_histogram_and_modes(distribution, FTC_segmentation(distribution))
incr = generate_decreasing_with_noise(30,0,10,0.5)
decr = generate_decreasing_with_noise(30,10,0,0.5)
distrib = np.concatenate((zeros,incr,decr,zeros))
plot_histogram_and_modes(distrib,FTC_segmentation(distrib))
"""
##test
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
im = r'./images_test/test6.jpeg'
from rbg_to_hsi_conversion import hsi  
im = hsi(im)
hue = im[:,0]
hist= histogramme(hue)
modes = FTC_segmentation(hist)
modes = np.array(modes, dtype=int)


hist_1 = np.histogram(hue, bins=180)[0]
modes1 = FTC_segmentation(hist_1)
modes1 = np.array(modes1, dtype=int)

plot_histogram_and_modes(hist, modes, file_path="histogram_modes.png")
plot_histogram_and_modes(hist_1, modes1)

