# coding: utf8
import cv2
import numpy as np

# returns refined filters
def guidedFilter(src, guide, r=45, eps=0.0001):
    src = np.array(src, np.float32)
    guide = np.array(guide, np.float32)
    src_pad = np.pad(src, ((r,r),(r,r),(0,0)), 'reflect')
    guide = np.pad(guide, ((r,r),(r,r),(0,0)), 'reflect')
    
    # Initialize a, b and output
    w = 2 * r + 1
    a_k = np.zeros(src_pad.shape[0:2], np.float32)
    b_k = np.zeros(src_pad.shape[0:2], np.float32)
    out = np.array(src, np.uint8, copy=True)
    
    # Calculate a and b by taking a window of size w * w
    for i in range(r, src_pad.shape[0]-r):
        for j in range(r, src_pad.shape[1]-r):
            # Initialize windows
            I = guide[i-r : i+r+1, j-r : j+r+1, 0]
            P = src_pad[i-r : i+r+1, j-r : j+r+1, 0]

            # Calculate each value in matrix a and b
            temp = np.dot(np.ndarray.flatten(I), np.ndarray.flatten(P))/(w*w)
            mu_k = np.mean(I)
            del_k = np.var(I)
            P_k_bar = np.mean(P)
            a_k[i,j] = (temp - mu_k * P_k_bar) / (del_k + eps)
            b_k[i,j] = P_k_bar - a_k[i,j] * mu_k

    # Mean of parameters in a and b due to multiple windows
    for i in range(r, src_pad.shape[0]-r):
        for j in range(r, src_pad.shape[1]-r):
            # Calculate mean
            a_k_bar = a_k[i-r : i+r+1, j-r : j+r+1].sum()/(w*w)
            b_k_bar = b_k[i-r : i+r+1, j-r : j+r+1].sum()/(w*w)
            # Calculate refined weights
            out[i-r,j-r] = np.round(a_k_bar * guide[i,j] + b_k_bar)

    return out


def fusion(img_1, img_2):
    # base and detail layers
    b1 = np.array(cv2.blur(img_1, (31,31)), np.int16)
    d1 = np.array(img_1, np.int16) - b1
    b2 = np.array(cv2.blur(img_2, (31,31)), np.int16)
    d2 = np.array(img_2, np.int16) - b2

    # saliency maps
    s1 = cv2.GaussianBlur(abs(cv2.Laplacian(img_1, cv2.CV_64F)), (5,5), 0)
    s2 = cv2.GaussianBlur(abs(cv2.Laplacian(img_2, cv2.CV_64F)), (5,5), 0)

    # weight maps
    p1 = np.zeros(img_1.shape, np.uint8)
    p2 = np.zeros(img_2.shape, np.uint8)
    p1[s1 >= s2] = 1
    p2[s2 > s1] = 1

    w1 = guidedFilter(p1, img_1, r=45, eps=0.3)
    w2 = guidedFilter(p2, img_2, r=7, eps=0.000001)

    # Fuse base and detail images using refined maps from guided filter
    bf = w1 * b1 + w2 * b2
    df = w1 * d1 + w2 * d2

    fused_img = np.array(bf+df, np.uint8)

    return fused_img

if __name__ == "__main__":
    # Read input images to be fused 
    img_1 = cv2.imread('a10_1.tif')
    img_2 = cv2.imread('a10_2.tif')
    fused_img = fusion(img_1, img_2)
    
    cv2.imshow('Final Fused Image', fused_img)