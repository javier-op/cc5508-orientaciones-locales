
import argparse
import matplotlib.pyplot as plt
import numpy as np
import orientation_histograms as oh
import pai_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots local orientations.')
    parser.add_argument('-image', type=str, help='Image to calculate local orientations.', required=True)
    parser.add_argument('-k', type=int, help='Grid dimensions.', required=True)
    args = parser.parse_args()
    image = pai_io.imread(args.image, as_gray = True)
    K = args.k
    print('Generating local orientations without bilinear interpolation')
    A1, R1 = oh.compute_local_orientations(image, K)
    print('Generating local orientations with bilinear interpolation')
    A2, R2 = oh.compute_local_orientations_bilinear(image, K)
    ys = np.arange(K)
    xs = np.arange(K)
    ys = np.floor(( (ys + 0.5) / K ) * image.shape[0])
    xs = np.floor(( (xs + 0.5) / K ) * image.shape[1]) 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].imshow(image, cmap = 'gray', vmax = 255, vmin = 0)
    axes[0].quiver(xs, ys, np.cos(A1)*R1, np.sin(A1)*R1, angles = 'xy', color = 'r')
    axes[0].axis('off')
    axes[1].imshow(image, cmap = 'gray', vmax = 255, vmin = 0)
    axes[1].quiver(xs, ys, np.cos(A2)*R2, np.sin(A2)*R2, angles = 'xy', color = 'r')
    axes[1].axis('off')
    fig.tight_layout()
    plt.show()
