import scipy.ndimage.filters as nd_filters
import numpy as np
from numpy import arctan2


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
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            p_prime = (i / image.shape[0]) * cell_size
            q_prime = (j / image.shape[1]) * cell_size
            l_pos = int(np.floor(p_prime - 0.5))
            r_pos = int(np.floor(p_prime + 0.5))
            n_pos = int(np.floor(q_prime - 0.5))
            s_pos = int(np.floor(q_prime + 0.5))
            if r_pos == cell_size:
                r_pos -= 1
            if s_pos == cell_size:
                s_pos -= 1
            dist_p = p_prime - np.floor(p_prime)
            if dist_p < 0.5:
                l_weight = 0.5 - dist_p
                r_weight = 1 - l_weight
            else:
                r_weight = dist_p - 0.5
                l_weight = 1 - r_weight
            dist_q = q_prime - np.floor(q_prime)
            if dist_q < 0.5:
                n_weight = 0.5 - dist_q
                s_weight = 1 - n_weight
            else:
                s_weight = dist_q - 0.5
                n_weight = 1 - s_weight
            Gx_local[l_pos, n_pos] += l_weight * n_weight * (np.square(gx[i, j]) - np.square(gy[i, j]))
            Gx_local[r_pos, n_pos] += r_weight * n_weight * (np.square(gx[i, j]) - np.square(gy[i, j]))
            Gx_local[l_pos, s_pos] += l_weight * s_weight * (np.square(gx[i, j]) - np.square(gy[i, j]))
            Gx_local[r_pos, s_pos] += r_weight * s_weight * (np.square(gx[i, j]) - np.square(gy[i, j]))
            Gy_local[l_pos, n_pos] += l_weight * n_weight * (2.0 * gx[i, j] * gy[i, j])
            Gy_local[r_pos, n_pos] += r_weight * n_weight * (2.0 * gx[i, j] * gy[i, j])
            Gy_local[l_pos, s_pos] += l_weight * s_weight * (2.0 * gx[i, j] * gy[i, j])
            Gy_local[r_pos, s_pos] += r_weight * s_weight * (2.0 * gx[i, j] * gy[i, j])
            r_local[l_pos, n_pos] += l_weight * n_weight * np.sqrt(np.square(gx[i, j]) + np.square(gy[i, j]))
            r_local[r_pos, n_pos] += r_weight * n_weight * np.sqrt(np.square(gx[i, j]) + np.square(gy[i, j]))
            r_local[l_pos, s_pos] += l_weight * s_weight * np.sqrt(np.square(gx[i, j]) + np.square(gy[i, j]))
            r_local[r_pos, s_pos] += r_weight * s_weight * np.sqrt(np.square(gx[i, j]) + np.square(gy[i, j]))
    local_ang = arctan2(Gy_local, Gx_local) * 0.5
    local_ang = local_ang + np.pi*0.5 # 0 <= ang  <= pi    
    return local_ang,  r_local

def compute_orientation_histogram_lineal(ang_local, r_local, L):
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
    