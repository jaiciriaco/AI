import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)


def load_and_downscale(path, scale_factor=2):
    img = cv.imread(path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if scale_factor != 1:
        h, w = gray.shape
        new_size = (w // scale_factor, h // scale_factor)
        img = cv.resize(img, new_size, interpolation=cv.INTER_LINEAR)
        gray = cv.resize(gray, new_size, interpolation=cv.INTER_LINEAR)

    return img, gray


def get_foreground_mask(gray, thresh_value=20):
    _, fg = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)
    return fg


def get_brain_mask(foreground):
    num_labels, labels = cv.connectedComponents(foreground)
    counts = np.bincount(labels.flatten())
    counts[0] = 0
    largest_label = counts.argmax()
    brain_mask = np.uint8(labels == largest_label) * 255
    return brain_mask


def create_watershed_seeds(brain_mask, dilate_iters=2, erode_iters=20, kernel_size=3):
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (kernel_size, kernel_size))

    sure_bg = cv.dilate(brain_mask, kernel, iterations=dilate_iters)
    sure_fg = cv.erode(brain_mask, kernel, iterations=erode_iters)
    unknown = cv.subtract(sure_bg, sure_fg)

    return sure_bg, sure_fg, unknown


def run_watershed(color_img, sure_fg, unknown):
    num_labels, labels = cv.connectedComponents(sure_fg)

    markers = labels + 1

    markers[unknown == 255] = 0

    markers_ws = cv.watershed(color_img, markers)
    return markers_ws


def get_inner_brain(brain_mask, iterations=4, kernel_size=5):
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner = cv.erode(brain_mask, kernel, iterations=iterations)
    return inner


def classify_lobes(markers_ws, inner_brain, min_overlap=0.7):
    h, w = markers_ws.shape
    inner_bool = inner_brain.astype(bool)

    label_stats = []
    for label in np.unique(markers_ws):
        if label <= 1:
            continue

        mask = (markers_ws == label)
        if not mask.any():
            continue

        inside = (mask & inner_bool).sum()
        total = mask.sum()
        overlap = inside / total

        if overlap < min_overlap:
            continue

        ys, xs = np.where(mask)
        mean_x = xs.mean()
        mean_y = ys.mean()
        label_stats.append((label, mean_x, mean_y))

    label_stats_sorted = sorted(label_stats, key=lambda t: t[2], reverse=True)

    temporals = label_stats_sorted[:2]
    central = label_stats_sorted[2:]

    if len(temporals) == 2:
        if temporals[0][1] < temporals[1][1]:
            left_label = temporals[0][0]
            right_label = temporals[1][0]
        else:
            left_label = temporals[1][0]
            right_label = temporals[0][0]
    elif len(temporals) == 1:
        left_label = right_label = temporals[0][0]
    else:
        left_label = right_label = None

    central_labels = [t[0] for t in central]

    return central_labels, left_label, right_label


def color_lobes(base_img, markers_ws, central_labels,
                left_label, right_label):
    result = cv.cvtColor(base_img, cv.COLOR_BGR2RGB)

    if central_labels:
        central_mask = np.isin(markers_ws, central_labels)
        result[central_mask] = [0, 0, 255]

    if left_label is not None:
        result[markers_ws == left_label] = [0, 255, 0]

    if right_label is not None:
        result[markers_ws == right_label] = [255, 0, 0]

    return result


if __name__ == "__main__":
    img, gray = load_and_downscale("data/brain1.png", scale_factor=2)

    fg_mask = get_foreground_mask(gray, thresh_value=20)
    brain_mask = get_brain_mask(fg_mask)

    sure_bg, sure_fg, unknown = create_watershed_seeds(
        brain_mask, dilate_iters=2, erode_iters=20, kernel_size=3)
    markers_ws = run_watershed(img, sure_fg, unknown)

    inner_brain = get_inner_brain(brain_mask, iterations=4, kernel_size=5)
    central_labels, left_label, right_label = classify_lobes(
        markers_ws, inner_brain, min_overlap=0.7)

    colored = color_lobes(img, markers_ws, central_labels,
                          left_label, right_label)

    plt.figure()
    plt.imshow(colored)
    plt.title("Central (Azul), Temporal.izq (verde), Temporal.der (Rojo)")
    plt.show()
