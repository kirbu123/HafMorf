from __future__ import annotations

import argparse
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.transform import radon


PDF_IMAGE_OBJECT_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj\s*(.*?)\s*stream\r?\n(.*?)\r?\nendstream", re.S)


@dataclass(frozen=True)
class PdfImage:
    image: np.ndarray
    width: int
    height: int
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deskew a scanned table document with a grayscale Radon transform "
            "and suppress table lines with grayscale morphology."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/target.Pdf"),
        help="Input PDF or image path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result"),
        help="Directory for result images and metadata.",
    )
    parser.add_argument(
        "--angle-range",
        type=float,
        default=15.0,
        help="Search deskew angles in [-angle-range, +angle-range] degrees.",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=0.1,
        help="Deskew angle search step in degrees.",
    )
    parser.add_argument(
        "--max-radon-dim",
        type=int,
        default=1100,
        help="Downsample the larger image side for Radon scoring.",
    )
    parser.add_argument(
        "--line-scale",
        type=float,
        default=0.035,
        help="Morphological line kernel length as fraction of image size.",
    )
    return parser.parse_args()


def load_input(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_first_pdf_image(path).image
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or unreadable input image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_first_pdf_image(path: Path) -> PdfImage:
    data = path.read_bytes()
    candidates: list[PdfImage] = []
    for match in PDF_IMAGE_OBJECT_RE.finditer(data):
        header = match.group(3)
        stream = match.group(4)
        if b"/Subtype /Image" not in header:
            continue
        width = parse_pdf_int(header, b"/Width")
        height = parse_pdf_int(header, b"/Height")
        bits = parse_pdf_int(header, b"/BitsPerComponent")
        colorspace = parse_pdf_name(header, b"/ColorSpace")
        image = decode_pdf_image_stream(stream, header, width, height, bits, colorspace)
        candidates.append(PdfImage(image=image, width=width, height=height, source=f"{path}:{match.group(1).decode()}"))
    if not candidates:
        raise ValueError(f"No embedded image streams were found in {path}")
    return max(candidates, key=lambda item: item.width * item.height)


def parse_pdf_int(header: bytes, key: bytes) -> int:
    match = re.search(re.escape(key) + rb"\s+(\d+)", header)
    if match is None:
        raise ValueError(f"Missing PDF image field {key.decode('latin1')}")
    return int(match.group(1))


def parse_pdf_name(header: bytes, key: bytes) -> bytes:
    match = re.search(re.escape(key) + rb"\s+(/[A-Za-z0-9]+)", header)
    if match is None:
        raise ValueError(f"Missing PDF image field {key.decode('latin1')}")
    return match.group(1)


def decode_pdf_image_stream(
    stream: bytes,
    header: bytes,
    width: int,
    height: int,
    bits_per_component: int,
    colorspace: bytes,
) -> np.ndarray:
    if bits_per_component != 8:
        raise ValueError(f"Only 8-bit PDF image streams are supported, got {bits_per_component}")
    if b"/FlateDecode" in header:
        raw = zlib.decompress(stream)
    else:
        raise ValueError("Only /FlateDecode PDF image streams are supported without extra dependencies")
    channels = 3 if colorspace == b"/DeviceRGB" else 1 if colorspace == b"/DeviceGray" else None
    if channels is None:
        raise ValueError(f"Unsupported PDF image colorspace: {colorspace.decode('latin1')}")
    expected = width * height * channels
    if len(raw) < expected:
        raise ValueError(f"PDF image stream is too short: got {len(raw)} bytes, expected {expected}")
    image = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, channels)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def to_gray_float(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def estimate_skew_angle(
    gray: np.ndarray,
    angle_range: float,
    angle_step: float,
    max_dim: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    work = gray
    scale = max(gray.shape) / float(max_dim)
    if scale > 1:
        work = cv2.resize(gray, (round(gray.shape[1] / scale), round(gray.shape[0] / scale)), interpolation=cv2.INTER_AREA)
    continuous_edges = sobel(work)
    continuous_edges = continuous_edges - float(np.min(continuous_edges))
    if float(np.max(continuous_edges)) > 0:
        continuous_edges = continuous_edges / float(np.max(continuous_edges))
    search = np.arange(90.0 - angle_range, 90.0 + angle_range + angle_step / 2.0, angle_step)
    sinogram = radon(continuous_edges, theta=search, circle=False, preserve_range=True)
    scores = np.var(sinogram, axis=0)
    best_projection_angle = float(search[int(np.argmax(scores))])
    skew_angle = best_projection_angle - 90.0
    return skew_angle, search, scores


def rotate_image(image: np.ndarray, angle_degrees: float, border_value: int = 255) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )


def remove_table_lines_grayscale(image: np.ndarray, line_scale: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    horizontal_len = max(15, int(round(image.shape[1] * line_scale)))
    vertical_len = max(15, int(round(image.shape[0] * line_scale)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))
    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.max(horizontal, vertical)
    cleaned_gray = 255 - cv2.subtract(inverted, lines)
    cleaned_rgb = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2RGB)
    return cleaned_rgb, lines


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_radon_plot(path: Path, angles: np.ndarray, scores: np.ndarray, skew_angle: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(angles - 90.0, scores, color="#255c99", linewidth=1.5)
    ax.axvline(skew_angle, color="#c2410c", linestyle="--", linewidth=1.2, label=f"skew={skew_angle:.2f} deg")
    ax.set_xlabel("deskew angle, degrees")
    ax.set_ylabel("Radon projection variance")
    ax.set_title("Grayscale Radon orientation score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_metadata(path: Path, input_path: Path, skew_angle: float, args: argparse.Namespace) -> None:
    path.write_text(
        "\n".join(
            [
                f"input={input_path}",
                f"skew_angle_degrees={skew_angle:.6f}",
                f"angle_range={args.angle_range}",
                f"angle_step={args.angle_step}",
                f"max_radon_dim={args.max_radon_dim}",
                f"line_scale={args.line_scale}",
                "binarization=false",
                "orientation_method=grayscale_radon_on_sobel_response",
                "line_removal_method=grayscale_morphological_opening_on_inverted_image",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_input(input_path)
    gray = to_gray_float(image)
    skew_angle, angles, scores = estimate_skew_angle(gray, args.angle_range, args.angle_step, args.max_radon_dim)
    deskewed = rotate_image(image, -skew_angle)
    cleaned, detected_lines = remove_table_lines_grayscale(deskewed, args.line_scale)

    save_image(output_dir / "01_input.png", image)
    save_image(output_dir / "02_oriented.png", deskewed)
    save_image(output_dir / "03_detected_lines.png", detected_lines)
    save_image(output_dir / "04_without_table_lines.png", cleaned)
    save_radon_plot(output_dir / "radon_score.png", angles, scores, skew_angle)
    write_metadata(output_dir / "metadata.txt", input_path, skew_angle, args)
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"skew_angle_degrees: {skew_angle:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
