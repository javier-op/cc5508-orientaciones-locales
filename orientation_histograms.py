import scipy.ndimage.filters as nd_filters
import numpy as np
from numpy import arctan2

def compute_orientation_histogram_basic(image, K):
    h = np.zeros(K, np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    ang = np.arctan2(gy,gx)    
    ang[ang < 0] = ang[ang < 0] + np.pi #sin signo    
    indx = np.round(K * ang / np.pi) 
    indx[indx ==  K] = 0
    for i in range(K):            
        h[i] = np.sum(indx == i)    
    return h    


def compute_orientation_histogram(image, K):
    h = np.zeros(K, np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    ang = np.arctan2(gy,gx)    
    ang[ang < 0] = ang[ang < 0] + np.pi #sin signo
    mag = np.sqrt(np.square(gy) + np.square(gx))    
    indx = np.round(K * ang / np.pi) 
    indx[indx ==  K] = 0
    for i in range(K):
        rows, cols = np.where(indx  == i)        
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)  #vector unitario    
    return h


def compute_local_orientations(image, cell_size):
    Gx_local = np.zeros((cell_size, cell_size), dtype = np.float32)
    Gy_local = np.zeros((cell_size, cell_size), dtype = np.float32)
    r_local = np.zeros((cell_size, cell_size), dtype = np.float32)    
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    mag = np.sqrt(np.square(gy) + np.square(gx))
    idx_rows, idx_cols = np.indices(image.shape)
    idx_grid_rows = np.floor(cell_size * idx_rows / image.shape[0])
    idx_grid_cols = np.floor(cell_size * idx_cols / image.shape[1])
    for p in np.arange(cell_size) :
        for q in np.arange(cell_size) :
            rows, cols = np.where((idx_grid_rows == p) & (idx_grid_cols == q))
            local_gx = gx[rows, cols]
            local_gy = gy[rows, cols]
            local_mag = mag[rows, cols]            
            Gx_local[p,q] = np.sum((np.square(local_gx) - np.square(local_gy))) 
            Gy_local[p,q] = np.sum(2.0*(local_gx*local_gy))
            r_local[p,q] = np.mean(local_mag)
    local_ang = arctan2(Gy_local, Gx_local) * 0.5
    local_ang = local_ang + np.pi*0.5 # 0 <= ang  <= pi    
    return local_ang,  r_local


def compute_local_orientations_bilinear(image, cell_size):
    Gx_local = np.zeros((cell_size, cell_size), dtype = np.float32)
    Gy_local = np.zeros((cell_size, cell_size), dtype = np.float32)
    r_local = np.zeros((cell_size, cell_size), dtype = np.float32)    
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    idx_rows, idx_cols = np.indices(image.shape)
    p_prime = (idx_rows / image.shape[0]) * cell_size
    q_prime = (idx_cols / image.shape[1]) * cell_size
    l_pos = np.floor(p_prime - 0.5)
    r_pos = np.floor(p_prime + 0.5)
    n_pos = np.floor(q_prime - 0.5)
    s_pos = np.floor(q_prime + 0.5)
    dist_p = p_prime - np.floor(p_prime)
    l_weight = np.where(dist_p < 0.5, 0.5 - dist_p, 0)
    r_weight = np.where(dist_p < 0.5, 1 - l_weight, 0)
    r_weight = np.where(dist_p >= 0.5, dist_p - 0.5, r_weight)
    l_weight = np.where(dist_p >= 0.5, 1 - r_weight, l_weight)
    dist_q = q_prime - np.floor(q_prime)
    n_weight = np.where(dist_q < 0.5, 0.5 - dist_q, 0)
    s_weight = np.where(dist_q < 0.5, 1 - n_weight, 0)
    s_weight = np.where(dist_q >= 0.5, dist_q - 0.5, s_weight)
    n_weight = np.where(dist_q >= 0.5, 1 - s_weight, n_weight)
    for p in np.arange(cell_size) :
        for q in np.arange(cell_size) :
            Gx_local[p, q] += np.sum(np.where((l_pos == p) & (n_pos == q), l_weight * n_weight * (np.square(gx) - np.square(gy)), 0))
            Gx_local[p, q] += np.sum(np.where((r_pos == p) & (n_pos == q), r_weight * n_weight * (np.square(gx) - np.square(gy)), 0))
            Gx_local[p, q] += np.sum(np.where((l_pos == p) & (s_pos == q), l_weight * s_weight * (np.square(gx) - np.square(gy)), 0))
            Gx_local[p, q] += np.sum(np.where((r_pos == p) & (s_pos == q), r_weight * s_weight * (np.square(gx) - np.square(gy)), 0))
            Gy_local[p, q] += np.sum(np.where((l_pos == p) & (n_pos == q), l_weight * n_weight * 2.0 * gx * gy, 0))
            Gy_local[p, q] += np.sum(np.where((r_pos == p) & (n_pos == q), r_weight * n_weight * 2.0 * gx * gy, 0))
            Gy_local[p, q] += np.sum(np.where((l_pos == p) & (s_pos == q), l_weight * s_weight * 2.0 * gx * gy, 0))
            Gy_local[p, q] += np.sum(np.where((r_pos == p) & (s_pos == q), r_weight * s_weight * 2.0 * gx * gy, 0))
            r_local[p, q] += np.sum(np.where((l_pos == p) & (n_pos == q), l_weight * n_weight * np.sqrt(np.square(gx) + np.square(gy)), 0))
            r_local[p, q] += np.sum(np.where((r_pos == p) & (n_pos == q), r_weight * n_weight * np.sqrt(np.square(gx) + np.square(gy)), 0))
            r_local[p, q] += np.sum(np.where((l_pos == p) & (s_pos == q), l_weight * s_weight * np.sqrt(np.square(gx) + np.square(gy)), 0))
            r_local[p, q] += np.sum(np.where((r_pos == p) & (s_pos == q), r_weight * s_weight * np.sqrt(np.square(gx) + np.square(gy)), 0))
    local_ang = arctan2(Gy_local, Gx_local) * 0.5
    local_ang = local_ang + np.pi*0.5 # 0 <= ang  <= pi    
    return local_ang,  r_local


def shelo(ang_local, r_local, L):
    h = np.zeros(L, np.float32)
    ang_local[ang_local < 0] = ang_local[ang_local < 0] + np.pi
    ang_prime = (L * ang_local) / np.pi
    l_pos = np.floor(ang_prime - 0.5)
    r_pos = np.floor(ang_prime + 0.5)
    for i in range(L):       
        h[i] += np.sum(np.where(l_pos == i, r_local, 0))
        h[i] += np.sum(np.where(r_pos == i, r_local, 0))
    h =  h / np.linalg.norm(h,2)  #vector unitario    
    return h
    