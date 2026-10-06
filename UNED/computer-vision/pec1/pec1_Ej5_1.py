import os
import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 8)

data_path = os.path.join("data", "threshold")

files = [
    "TH_noisy2.png",
    "TH_TE.gif",
    "TH_scanTable.png"
]


def load_gray(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img


def show_global_threshold(imgs, names):
    for img, name in zip(imgs, names):
        _, th_fixed = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

        _, th_otsu = cv.threshold(
            img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        fig, axes = plt.subplots(1, 3)
        fig.suptitle(f"Umbralización global - {name}")

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(th_fixed, cmap="gray")
        axes[1].set_title("Global fijo T=128")
        axes[1].axis("off")

        axes[2].imshow(th_otsu, cmap="gray")
        axes[2].set_title(f"Global Otsu (T={int(_):d})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def explore_adaptive_threshold(img, name, invert=True):
    if invert:
        img_proc = 255 - img
    else:
        img_proc = img.copy()

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f"Adaptive threshold (MEAN_C) - {name} (invert={invert})")

    W_list = [5, 11, 25]
    C_list = [-5, 0, 5]

    for i, W in enumerate(W_list):
        for j, C in enumerate(C_list):
            th = cv.adaptiveThreshold(
                img_proc,
                255,
                cv.ADAPTIVE_THRESH_MEAN_C,
                cv.THRESH_BINARY,
                W,
                C
            )
            ax = axes[i, j]
            ax.imshow(th, cmap="gray")
            ax.set_title(f"W={W}, C={C}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    images = []
    for fname in files:
        path = os.path.join(data_path, fname)
        img = load_gray(path)
        images.append(img)

    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    fig.suptitle("Imágenes originales (5.1)")
    for ax, img, name in zip(axes, images, files):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    show_global_threshold(images, files)

    for img, name in zip(images, files):
        explore_adaptive_threshold(img, name, invert=True)


if __name__ == "__main__":
    main()
