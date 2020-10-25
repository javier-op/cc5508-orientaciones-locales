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
        if distance <= 0.01:
            continue
        results.append((filename, distance))
        counter += 1
    return results


def get_fingerprint_to_class_map():
    fingerprint_to_class_map = {}
    root_path = Path('./fingerprints').glob('*')
    for class_path in root_path:
        for fingerprint in class_path.glob('*'):
            fingerprint_to_class_map[str(fingerprint)] = str(class_path) 
    return fingerprint_to_class_map


def calculate_recover_error(local_orientation_function):
    finger_histograms = calculate_all_fingerprints_histograms(local_orientation_function, args.k, args.l)
    fingerprint_to_class_map = get_fingerprint_to_class_map()
    recover_errors = []
    for current_filename in finger_histograms:
        current_results = get_top_5_histogram_matches(finger_histograms[current_filename], finger_histograms)
        recover_error = 0
        for result_filename, distance in current_results:
            if fingerprint_to_class_map[current_filename] != fingerprint_to_class_map[result_filename]:
                recover_error += 1.
        recover_errors.append(recover_error)
    return np.mean(recover_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds most similar fingerprints.')
    parser.add_argument('--image', type=str, help='Input image..', required=True)
    parser.add_argument('-k', type=int, help='Grid dimensions for local orientations.', required=True)
    parser.add_argument('-l', type=int, help='Array dimensions for histogram.', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--non-bilinear', help='Calculate local orientations without bilinear interpolation.', action='store_true')
    mode.add_argument('--eval', help='Performance evaluation mode.', action='store_true')
    args = parser.parse_args()
    image = pai_io.imread(args.image, as_gray = True)

    if args.eval:
        recover_error_regular = calculate_recover_error(oh.compute_local_orientations)
        recover_error_bilinear = calculate_recover_error(oh.compute_local_orientations_bilinear)
        print('Recover error without bilinear interpolation: ' + str(recover_error_regular))
        print('Recover error with bilinear interpolation: ' + str(recover_error_bilinear))
    else:
        if args.non_bilinear:
            local_orientation_function = oh.compute_local_orientations
        else:
            local_orientation_function = oh.compute_local_orientations_bilinear
        finger_histograms = calculate_all_fingerprints_histograms(local_orientation_function, args.k, args.l)
        ang_local, r_local = local_orientation_function(image, args.k)
        input_histogram = oh.compute_orientation_histogram_lineal(ang_local, r_local, args.l)
        results = get_top_5_histogram_matches(input_histogram, finger_histograms)
        print(results)
