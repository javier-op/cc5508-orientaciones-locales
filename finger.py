import argparse
import heapq
import matplotlib.pyplot as plt
import numpy as np
import orientation_histograms as oh
import pai_io
from pathlib import Path
import pickle

def calculate_all_fingerprints_histograms(K, L):
    try:
        with open('finger_orientations.pkl', 'rb') as input:
            finger_orientations = pickle.load(input)
    except (IOError, pickle.PickleError):
        finger_orientations = {}
    if K in finger_orientations:
        finger_orientations_k = finger_orientations[K]
    else:
        finger_orientations_k = {}
        pathlist = Path('./fingerprints').rglob('*.png')
        for path in pathlist:
            image = pai_io.imread(str(path), as_gray = True)
            ang_local, r_local = oh.compute_local_orientations_bilinear(image, K)
            finger_orientations_k[str(path)] = (ang_local, r_local)
        finger_orientations[K] = finger_orientations_k
        with open('finger_orientations.pkl', 'wb') as output:
            pickle.dump(finger_orientations, output, pickle.HIGHEST_PROTOCOL)
    finger_histograms = {}
    for filename in finger_orientations_k:
        ang_local, r_local = finger_orientations_k[filename]
        finger_histograms[filename] = oh.shelo(ang_local, r_local, L)
    return finger_histograms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds most similar fingerprints.')
    parser.add_argument('-image', type=str, help='Input image..', required=True)
    parser.add_argument('-k', type=int, help='Grid dimensions for local orientations.', required=True)
    parser.add_argument('-l', type=int, help='Array dimensions for histogram.', required=True)
    args = parser.parse_args()
    finger_histograms = calculate_all_fingerprints_histograms(args.k, args.l)
    image = pai_io.imread(args.image, as_gray = True)
    ang_local, r_local = oh.compute_local_orientations_bilinear(image, args.k)
    input_histogram = oh.shelo(ang_local, r_local, args.l)
    heap = []
    for filename in finger_histograms:
        finger_histogram = finger_histograms[filename]
        distance = np.sqrt(np.sum(np.square(input_histogram - finger_histogram)))
        heapq.heappush(heap, (distance, filename))
    results = []
    for i in range(5):
        distance, filename = heapq.heappop(heap)
        results.append((filename, distance))
    print('Done.')
