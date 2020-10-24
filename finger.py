import argparse
import heapq
import matplotlib.pyplot as plt
import numpy as np
import orientation_histograms as oh
import pai_io
from pathlib import Path
import pickle


def calculate_all_fingerprints_histograms(local_orientation_function, K, L):
    try:
        with open(local_orientation_function.__name__+'.pkl', 'rb') as input:
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
            ang_local, r_local = local_orientation_function(image, K)
            finger_orientations_k[str(path)] = (ang_local, r_local)
        finger_orientations[K] = finger_orientations_k
        with open(local_orientation_function.__name__+'.pkl', 'wb') as output:
            pickle.dump(finger_orientations, output, pickle.HIGHEST_PROTOCOL)
    finger_histograms = {}
    for filename in finger_orientations_k:
        ang_local, r_local = finger_orientations_k[filename]
        finger_histograms[filename] = oh.compute_orientation_histogram_lineal(ang_local, r_local, L)
    return finger_histograms


def get_top_5_histogram_matches(input_histogram, finger_histograms):
    heap = []
    for filename in finger_histograms:
        finger_histogram = finger_histograms[filename]
        distance = np.sqrt(np.sum(np.square(input_histogram - finger_histogram)))
        heapq.heappush(heap, (distance, filename))
    results = []
    counter = 0
    while counter < 5:
        distance, filename = heapq.heappop(heap)
        if distance == 0:
            continue
        results.append((filename, distance))
        counter += 1
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds most similar fingerprints.')
    parser.add_argument('-image', type=str, help='Input image..', required=True)
    parser.add_argument('-k', type=int, help='Grid dimensions for local orientations.', required=True)
    parser.add_argument('-l', type=int, help='Array dimensions for histogram.', required=True)
    args = parser.parse_args()
    image = pai_io.imread(args.image, as_gray = True)

    finger_histograms = calculate_all_fingerprints_histograms(oh.compute_local_orientations, args.k, args.l)
    ang_local, r_local = oh.compute_local_orientations(image, args.k)
    input_histogram = oh.compute_orientation_histogram_lineal(ang_local, r_local, args.l)
    results_regular = get_top_5_histogram_matches(input_histogram, finger_histograms)

    finger_histograms_b = calculate_all_fingerprints_histograms(oh.compute_local_orientations_bilinear, args.k, args.l)
    ang_local_b, r_local_b = oh.compute_local_orientations_bilinear(image, args.k)
    input_histogram_b = oh.compute_orientation_histogram_lineal(ang_local_b, r_local_b, args.l)
    results_bilineal = get_top_5_histogram_matches(input_histogram_b, finger_histograms_b)

    print(results_regular)
    print(results_bilineal)
