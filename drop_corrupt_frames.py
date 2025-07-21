import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import os

def handle_sigint(sig, frame):
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)


def detect_local_vertical_split_strict(img, search_frac=0.45, diff_thresh=25.0, row_thresh=12.0, row_ratio_thresh=0.5):
    start = time.perf_counter()
    h, w = img.shape[:2]
    x0, x1 = int(w*search_frac), int(w*(1-search_frac))

    best_x, best_diff = x0, 0.0
    for x in range(x0, x1):
        d = np.linalg.norm(img[:, x].astype(np.int16) - img[:, x+1].astype(np.int16), axis=1)
        mean_diff = float(d.mean())
        if mean_diff > best_diff:
            best_diff, best_x = mean_diff, x

    d_best = np.linalg.norm(img[:, best_x].astype(np.int16) - img[:, best_x+1].astype(np.int16), axis=1)
    row_ratio = np.count_nonzero(d_best >= row_thresh) / float(h)

    is_corrupt = (best_diff > diff_thresh) and (row_ratio >= row_ratio_thresh)

    plot_column_differences(is_corrupt, d_best, best_x, row_thresh=row_thresh, row_ratio=row_ratio)

    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000.0
    return is_corrupt, best_x, best_diff, row_ratio, elapsed_ms


def draw_split(img, split_x, is_corrupt, diff, row_ratio):
    out = img.copy()
    if not is_corrupt:
        return out

    cv2.line(out, (split_x, 0), (split_x, out.shape[0]), (0, 0, 255), 2)
    label = f"CORRUPT (x={split_x}, DIFF={diff:.1f}, rows={row_ratio:.2f})"
    cv2.putText(out, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def plot_column_differences(is_corrupt, diffs, column_index, row_thresh=None, row_ratio=None):
    h = len(diffs)
    rows = range(h)

    plt.figure(figsize=(8, 4))
    plt.plot(rows, diffs, linewidth=1, label="Pixel Diff")
    plt.title(f"Pixel Differences at Column {column_index}")
    plt.ylabel("Pixel Difference")
    plt.xlabel("Row Index")

    if row_thresh is not None:
        plt.axhline(y=row_thresh, color='red', linestyle='--', linewidth=1, label=f"row_thresh={row_thresh}")

    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", f"column_diff_{column_index}_RR_{row_ratio:.3f}_CorruptionStatus_{is_corrupt}.png")
    plt.savefig(out_path, dpi=600)
    plt.close()


if __name__ == "__main__":
    tests = [
        ("Good Image", "source_images/github_issue_good_pic.png"),
        ("Bad Image", "source_images/github_issue_bad_pic.png"),
        ("Aristurtle Good Image", "source_images/Aristurtle_good.png"),
        ("Aristurtle Bad Image", "source_images/Aristurtle_bad.png"),
        ("Aristurtle Good Image 2", "source_images/Aristurtle_good_2.png"),
        ("Aristurtle Bad Image 2", "source_images/Aristurtle_bad_2.png"),
        ("Aristurtle Good Image 3", "source_images/Aristurtle_good_3.png"),
        ("Aristurtle Bad Image 3", "source_images/Aristurtle_bad_3.png"),
        ("Aristurtle Good Image 4", "source_images/Aristurtle_good_4.png"),
        ("Aristurtle Bad Image 4", "source_images/Aristurtle_bad_4.png"),
        ("Aristurtle Good Image 5", "source_images/Aristurtle_good_5.png"),
        ("Aristurtle Bad Image 5", "source_images/Aristurtle_bad_5.png"),
        ("Aristurtle Good Image 6", "source_images/Aristurtle_good_6.png"),
        ("Aristurtle Bad Image 6", "source_images/Aristurtle_bad_6.png"),
        ("Aristurtle Good Image 7", "source_images/Aristurtle_good_7.png"),
        ("Aristurtle Bad Image 7", "source_images/Aristurtle_bad_7.png"),
    ]

    for name, path in tests:
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot load '{path}'")
            continue

        corrupt, x, diff, rr, t_ms = detect_local_vertical_split_strict(img)
        print(f"{name}: corrupt={corrupt}, x={x}, DIFF={diff:.1f}, rows={rr:.2f}, time={t_ms:.2f} ms")
        disp = draw_split(img, x, corrupt, diff, rr)
        status_str = "corrupt" if corrupt else "clean"
        os.makedirs("annotated_images", exist_ok=True)
        out_path = os.path.join("annotated_images", f"annotated_{name.replace(' ', '_')}_{status_str}.png")
        cv2.imwrite(out_path, disp)
