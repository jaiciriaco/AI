import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 8)

img = cv.imread("data/brain1.png", cv.IMREAD_GRAYSCALE)

print("Tamaño de la imagen:", img.shape)

plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Imagen original - brain1.png")
plt.axis("off")
plt.show()

Z = img.reshape((-1, 1))
Z = np.float32(Z)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)

K_list = [3, 4, 5]
segmented_images = []

for K in K_list:
    print(f"\n=== Ejecutando k-means con K = {K} ===")

    compactness, labels, centers = cv.kmeans(
        Z, K, None, criteria, 5, cv.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    sorted_idx = np.argsort(centers.flatten())
    centers_sorted = centers[sorted_idx]

    labels_sorted = np.zeros_like(labels)
    for new_label, old_label in enumerate(sorted_idx):
        labels_sorted[labels == old_label] = new_label

    segm = centers_sorted[labels_sorted.flatten()]
    segm = segm.reshape(img.shape)
    segmented_images.append((K, segm))

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(f"Segmentación k-means con K = {K}")

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(segm, cmap="nipy_spectral")
    axes[1].set_title(f"Segmentación (K={K})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(1, len(segmented_images),
                         figsize=(5 * len(segmented_images), 5))
fig.suptitle("Comparación de k-means para distintos K")

for ax, (K, segm) in zip(axes, segmented_images):
    ax.imshow(segm, cmap="nipy_spectral")
    ax.set_title(f"K = {K}")
    ax.axis("off")

plt.tight_layout()
plt.show()
