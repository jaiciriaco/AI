import os
import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 8)

brain_path = os.path.join("data/jardin1.png")
plane_path = os.path.join("data/avion1.png")


def segment_meanshift_brain():
    img_gray = cv.imread(brain_path, cv.IMREAD_GRAYSCALE)

    img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

    sp_values = [3, 15, 30]
    sr_values = [10, 50, 100]

    fig, axes = plt.subplots(len(sp_values), len(sr_values),
                             figsize=(4*len(sr_values), 4*len(sp_values)))
    fig.suptitle("brain1 - pyrMeanShiftFiltering (segmentación por suavizado)")

    for i, sp in enumerate(sp_values):
        for j, sr in enumerate(sr_values):
            ms = cv.pyrMeanShiftFiltering(img_color, sp, sr, maxLevel=1)
            ms_gray = cv.cvtColor(ms, cv.COLOR_BGR2GRAY)
            _, seg = cv.threshold(
                ms_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )

            ax = axes[i, j]
            ax.imshow(seg, cmap="gray")
            ax.set_title(f"sp={sp}, sr={sr}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def segment_meanshift_plane():
    img = cv.imread(plane_path)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    sp_values = [5, 20, 40]
    sr_values = [20, 60, 120]

    fig, axes = plt.subplots(len(sp_values), len(sr_values),
                             figsize=(4*len(sr_values), 4*len(sp_values)))
    fig.suptitle(
        "avion1 - pyrMeanShiftFiltering (segmentación por color+espacio)")

    for i, sp in enumerate(sp_values):
        for j, sr in enumerate(sr_values):
            img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            ms = cv.pyrMeanShiftFiltering(img_bgr, sp, sr, maxLevel=1)
            ms_rgb = cv.cvtColor(ms, cv.COLOR_BGR2RGB)

            ax = axes[i, j]
            ax.imshow(ms_rgb)
            ax.set_title(f"sp={sp}, sr={sr}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    brain = cv.imread(brain_path, cv.IMREAD_GRAYSCALE)
    plane = cv.imread(plane_path)
    plane_rgb = cv.cvtColor(plane, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(brain, cmap="gray")
    axes[0].set_title("brain1.png (original)")
    axes[0].axis("off")

    axes[1].imshow(plane_rgb)
    axes[1].set_title("avion1.png (original)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    segment_meanshift_brain()
    segment_meanshift_plane()


if __name__ == "__main__":
    main()
