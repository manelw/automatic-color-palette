import numpy as np 
import matplotlib
import scipy
import scipy.linalg
import scipy.signal
import scipy.stats
from matplotlib import pyplot as plt
import cv2 as cv

def grenander_down(h) : 
    gren = np.copy(h)
    n = len(h)
    for j in range(n) :
        for i in range(n-1) : 
            if (gren[i] < gren[i+1]) :
                gren[i] = (gren[i] + gren[i+1])/2
                gren[i+1] = gren[i]
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
        for i in range(n-1) : 
            if (gren[i] > gren[i+1]) :
                gren[i] = (gren[i] + gren[i+1])/2
                gren[i+1] = gren[i]
    return gren


def plot_histogram_and_grenander_up(h):
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
    plt.plot(grenander, label='Grenander Down', color='orange')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Grenander Down Estimator')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def generate_decreasing_with_noise(size, start, end, noise_level):
    decreasing = np.linspace(start, end, size)
    noise = np.random.normal(0, noise_level, size)
    return decreasing + noise

decreasing_with_noise = generate_decreasing_with_noise(100, 10, 0, 0.5)
#plot_histogram_and_grenander_down(decreasing_with_noise)

increasing_with_noise = generate_decreasing_with_noise(100,0,10,0.5)
#plot_histogram_and_grenander_up(increasing_with_noise)

def binomial_sum(N,r,p) :
    return scipy.stats.binom.sf(r-1,N,p)

def NFA(r_bar,L,N,r) :
    p = np.sum(r_bar)
    #print(p)
    #print(r)
    return L*(L+1)/2*binomial_sum(N,N*r,p) if r >= p else L*(L+1)/2*binomial_sum(N,N*(1-r),1-p)

def meaningful_rej_dec(r,L,N,h) : 
    r_bar = grenander_down(1/N*h)
    print(r_bar)
    print(NFA(r_bar,L,N,r))
    return NFA(r_bar,L,N,r) < 1/2

def decr_hyp(a,b,h,L,N) : 
    r = 1/N*np.sum(h[a:b])
    if meaningful_rej_dec(r,L,N,h) :
        return False    
    return True

def meaningful_rej_incr(r,L,N,h) : 
    r_bar = grenander_up(1/N*h)
    print(NFA(r_bar,L,N,r))
    return NFA(r_bar,L,N,r) < 1/2
        
def incr_hyp(a,b,h,L,N) : 
    r = 1/N*np.sum(h[a:b])
    if meaningful_rej_incr(r,L,N,h) :
        return False    
    return True

def unimodal(a,b,h) : 
    L = len(h)
    N = np.sum(h)
    h = np.array(h)
    for c in range(a+1,b-1) : 
        c = int(c)
        print(c)
        if incr_hyp(a,c,h[a:c],L,N) and decr_hyp(c,b,h[c:b],L,N) : 
            return True
    return False

