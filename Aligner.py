"""
Simple GUI to create homographies between image modalities

The input folder must contain pairs with the same base name + their assigned suffix
(example: scene_rgb.jpg and scene_thermal.jpg) 

For each correspondence, click RGB first, then thermal.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg",}
POINT_COUNT = 5
MAX_IMAGE_SIZE = (700, 600)
WINDOW_NAME = "Image pair alignment"


def discover_pairs(
    folder: Path, RGB_suffix: str, IR_suffix: str
) -> list[tuple[str, Path, Path]]:
    """Find matching image files by their names"""
    if not RGB_suffix or not IR_suffix:
        raise ValueError("Both image modality suffixes must be assigned.")
    if RGB_suffix.casefold() == IR_suffix.casefold():
        raise ValueError("Image modality suffixes must be different.")

    grouped: dict[str, dict[str, Path]] = {}
    suffixes = {
        RGB_suffix.casefold(): "RGB",
        IR_suffix.casefold(): "IR",
    }

    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        folded_stem = stem.casefold()
        for suffix, role in suffixes.items():
            if folded_stem.endswith(suffix):
                base_name = stem[: -len(suffix)]
                if not base_name:
                    print(f"Skipping {path.name}: filename has no pair name.")
                    break
                pair = grouped.setdefault(base_name.casefold(), {})
                if role in pair:
                    raise ValueError(
                        f"Multiple {role} images found for pair "
                        f"{base_name!r}: {pair[role].name} and {path.name}."
                    )
                pair[role] = path
                break

    pairs: list[tuple[str, Path, Path]] = []
    for pair in grouped.values():
        if "RGB" not in pair:
            print(f"Skipping unmatched image: {pair['IR'].name}")
            continue
        if "IR" not in pair:
            print(f"Skipping unmatched image: {pair['RGB'].name}")
            continue
        RGB_path = pair["RGB"]
        IR_path = pair["IR"]
        RGB_suffix_length = len(RGB_suffix)
        name = RGB_path.stem[:-RGB_suffix_length]
        pairs.append((name, RGB_path, IR_path))

    return sorted(pairs, key=lambda pair: pair[0].casefold())


def fit_image(image: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    max_width, max_height = max_size
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


class PointAnnotator:
    def __init__(
        self, RGB: np.ndarray, IR: np.ndarray, pair_name: str, pair_index: int, total_pairs: int
    ) -> None:
        self.RGB = RGB
        self.IR = IR
        self.pair_name = pair_name
        self.pair_index = pair_index
        self.total_pairs = total_pairs
        self.RGB_view = fit_image(RGB, MAX_IMAGE_SIZE)
        self.IR_view = fit_image(IR, MAX_IMAGE_SIZE)
        self.RGB_points: list[tuple[float, float]] = []
        self.IR_points: list[tuple[float, float]] = []
        self.message = ""
        self.quit_requested = False
        self.skip_requested = False
        self.homography: np.ndarray | None = None
        self.aligned: np.ndarray | None = None
        self.overlay: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.RGB_offset = (0, 80)
        self.IR_offset = (
            self.RGB_view.shape[1] + 20,
            80,
        )
        self.canvas = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_mouse(
        self, event: int, x: int, y: int, flags: int, parameter: object
    ) -> None:
        del flags, parameter
        if event != cv2.EVENT_LBUTTONDOWN or self.homography is not None:
            return

        if len(self.RGB_points) == len(self.IR_points):
            image = self.RGB
            view = self.RGB_view
            offset_x, offset_y = self.RGB_offset
            destination = self.RGB_points
            role = "RGB"
        else:
            image = self.IR
            view = self.IR_view
            offset_x, offset_y = self.IR_offset
            destination = self.IR_points
            role = "thermal"

        local_x = x - offset_x
        local_y = y - offset_y
        view_height, view_width = view.shape[:2]
        if not (0 <= local_x < view_width and 0 <= local_y < view_height):
            return

        image_height, image_width = image.shape[:2]
        destination.append(
            (
                local_x * image_width / view_width,
                local_y * image_height / view_height,
            )
        )
        self.message = f"Point {len(destination)}: {role} selected."
        if len(self.RGB_points) == POINT_COUNT and len(
            self.IR_points
        ) == POINT_COUNT:
            self._calculate_homography()

    def _calculate_homography(self) -> None:
        source_points = np.asarray(self.IR_points, dtype=np.float32)
        destination_points = np.asarray(self.RGB_points, dtype=np.float32)
        homography, _ = cv2.findHomography(
            source_points, destination_points, cv2.RANSAC, 5.0
        )
        if homography is None or not np.isfinite(homography).all():
            self.message = (
                "Could not calculate a homography. Press U to undo or R to retry."
            )
            self.RGB_points.pop()
            self.IR_points.pop()
            return

        RGB_height, RGB_width = self.RGB.shape[:2]
        aligned = cv2.warpPerspective(
            self.IR, homography, (RGB_width, RGB_height)
        )
        source_mask = np.full(self.IR.shape[:2], 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            source_mask,
            homography,
            (RGB_width, RGB_height),
            flags=cv2.INTER_NEAREST,
        )
        overlay = cv2.addWeighted(self.RGB, 0.5, aligned, 0.5, 0)
        overlay[valid_mask == 0] = self.RGB[valid_mask == 0]
        self.homography = homography
        self.aligned = aligned
        self.overlay = overlay
        self.valid_mask = valid_mask
        self.message = "Review alignment: A accept | R redo | S skip | Q quit"

    #----------------------
    # Canvas visualizations
    #----------------------

    def _draw_annotation(self) -> np.ndarray:
        gap = 20
        height = max(
            self.RGB_view.shape[0], self.IR_view.shape[0]
        ) + self.RGB_offset[1]
        width = (
            self.RGB_view.shape[1]
            + gap
            + self.IR_view.shape[1]
        )
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        ref_x, ref_y = self.RGB_offset
        mov_x, mov_y = self.IR_offset
        ref_h, ref_w = self.RGB_view.shape[:2]
        mov_h, mov_w = self.IR_view.shape[:2]
        canvas[ref_y : ref_y + ref_h, ref_x : ref_x + ref_w] = (
            self.RGB_view
        )
        canvas[mov_y : mov_y + mov_h, mov_x : mov_x + mov_w] = (
            self.IR_view
        )

        self._draw_points(canvas, self.RGB_points, self.RGB, ref_x, ref_y)
        self._draw_points(canvas, self.IR_points, self.IR, mov_x, mov_y)
        cv2.putText(
            canvas,
            f"{self.pair_name} | Pairs Completed: {self.pair_index}/{self.total_pairs} |"
            f" Points: {len(self.IR_points)}/{POINT_COUNT}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Click RGB then matching thermal point | U undo | R reset | "
            "S skip | Q quit",
            (10, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if self.message:
            cv2.putText(
                canvas,
                self.message,
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_points(
        self,
        canvas: np.ndarray,
        points: list[tuple[float, float]],
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> None:
        image_height, image_width = image.shape[:2]
        view_height, view_width = (
            self.RGB_view.shape[:2]
            if image is self.RGB
            else self.IR_view.shape[:2]
        )
        for index, (point_x, point_y) in enumerate(points):
            x = offset_x + round(point_x * view_width / image_width)
            y = offset_y + round(point_y * view_height / image_height)
            cv2.circle(canvas, (x, y), 8, 0, -1)
            cv2.circle(canvas, (x, y), 5, (255,255,255), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                0,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                str(index + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                2,
                cv2.LINE_AA,
            )

    def _draw_review(self) -> np.ndarray:
        assert self.aligned is not None
        assert self.overlay is not None
        images = [
            ("RGB", self.RGB),
            ("Warped thermal", self.aligned),
            ("50/50 overlay", self.overlay),
        ]
        views = [(label, fit_image(image, (400, 650))) for label, image in images]
        gap = 12
        header_height = 72
        height = max(view.shape[0] for _, view in views) + header_height
        width = sum(view.shape[1] for _, view in views) + gap * len(views)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        offset_x = 0
        for label, view in views:
            view_height, view_width = view.shape[:2]
            canvas[
                header_height : header_height + view_height,
                offset_x : offset_x + view_width,
            ] = view
            cv2.putText(
                canvas,
                label,
                (offset_x + 5, header_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            offset_x += view_width + gap
        cv2.putText(
            canvas,
            f"{self.pair_name} | A accept | R redo | S skip | Q quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            self.message,
            (10, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def run(self) -> str:
        while True:
            if self.quit_requested:
                return "quit"
            if self.skip_requested:
                return "skip"

            if self.homography is None:
                self.canvas = self._draw_annotation()
            else:
                self.canvas = self._draw_review()
            cv2.imshow(WINDOW_NAME, self.canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                self.quit_requested = True
            elif key == ord("s"):
                self.pair_index += 1
                self.skip_requested = True
            elif key == ord("r"):
                self._reset()
            elif key == ord("u") and self.homography is None:
                self._undo()
            elif key == ord("a") and self.homography is not None:
                self.pair_index += 1
                return "accept"

    def _reset(self) -> None:
        self.RGB_points.clear()
        self.IR_points.clear()
        self.homography = None
        self.aligned = None
        self.overlay = None
        self.valid_mask = None
        self.message = "Points cleared."

    def _undo(self) -> None:
        if len(self.RGB_points) > len(self.IR_points):
            self.RGB_points.pop()
        elif self.RGB_points:
            self.RGB_points.pop()
            self.IR_points.pop()
        self.message = "Last point undone."

# Save homography and aligned images 
def save_result(
    output_dir: Path,
    pair_name: str,
    homography: np.ndarray,
    aligned: np.ndarray,
    overlay: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"{pair_name}_aligned.png": aligned,
        f"{pair_name}_overlay.png": overlay,
        f"{pair_name}_valid_mask.png": valid_mask,
    }
    for filename, image in outputs.items():
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write output image: {output_path}")
    np.save(output_dir / f"{pair_name}_homography.npy", homography)
    np.savetxt(
        output_dir / f"{pair_name}_homography.txt",
        homography,
        fmt="%.10g",
    )

# CLI parsing (all optional)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually align RGB/thermal image pairs with five point pairs."
    )
    parser.add_argument(
        "--input-folder", 
        type=Path, 
        default=Path("input"),
        help="Folder containing image pairs (default: ./input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Folder for approved alignments (default: ./output)",
    )
    parser.add_argument(
        "--RGB-suffix",
        default="_norm",
        help="Suffix before the extension for the RGB image (default: _norm)",
    )
    parser.add_argument(
        "--IR-suffix",
        default="_flir",
        help="Suffix before the extension for the IR image (default: _flir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder: Path = args.input_folder
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    pairs = discover_pairs(
        input_folder, args.RGB_suffix, args.IR_suffix
    )
    if not pairs:
        raise FileNotFoundError(
            f"No matching image pairs found in {input_folder}. "
            f"Expected pairs of *{args.RGB_suffix} and *{args.IR_suffix} "
            "image files."
        )

    # Check that image has not already been aligned (exists in output folder)
    output_folder: Path = args.output_dir
    if output_folder.is_dir():
        for outP in output_folder.iterdir():
            for (pair_name, RGB_path, IR_path) in (pairs):
                if outP.stem.removesuffix("_aligned") == pair_name:
                    pairs.remove((pair_name, RGB_path, IR_path))
                    print(
                        f"Filename \'{pair_name}\' exists in the input and output folder. "
                        f"Skipping {pair_name} files in input..."
                        )
                    

        

    print(f"Found {len(pairs)} valid image pair(s).")
    try:
        for index, (pair_name, RGB_path, IR_path) in enumerate(pairs, 1):
            RGB = cv2.imread(str(RGB_path), cv2.IMREAD_COLOR)
            IR = cv2.imread(str(IR_path), cv2.IMREAD_COLOR)
            if RGB is None:
                raise OSError(f"Could not read image: {RGB_path}")
            if IR is None:
                raise OSError(f"Could not read image: {IR_path}")

            print(
                f"[{index}/{len(pairs)}] {RGB_path.name} + "
                f"{IR_path.name}"
            )
            annotator = PointAnnotator(RGB, IR, pair_name, index, len(pairs))
            result = annotator.run()
            if result == "accept":
                assert annotator.homography is not None
                assert annotator.aligned is not None
                assert annotator.overlay is not None
                assert annotator.valid_mask is not None
                save_result(
                    args.output_dir,
                    pair_name,
                    annotator.homography,
                    annotator.aligned,
                    annotator.overlay,
                    annotator.valid_mask,
                )
                print(f"Saved approved alignment for {pair_name!r}.")
            elif result == "skip":
                print(f"Skipped {pair_name!r}.")
            else:
                print("Stopping.")
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
