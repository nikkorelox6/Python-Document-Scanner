# High-Quality AI Document Scanner

Converts a photograph of a paper document into an image that closely
resembles a page produced by a professional flatbed scanner: perspective
corrected, straightened, shadow- and color-cast-free, denoised, and cropped.

Quality is prioritized over speed. The pipeline is classical computer
vision (OpenCV / NumPy), not a trained model, so it has no external weights
to download and works fully offline.

## Pipeline

Each stage is an independent class/function, so any one of them can later
be swapped out (e.g. for a learned detector) without touching the rest:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | `DocumentLocalizer` | Finds the page boundary, its 4 corners, a confidence score, and whether the page is flat or curved |
| 2 | `GeometricReconstructor` | 4-point perspective transform, plus an optional thin-plate-spline correction for curled/folded edges |
| 3 | `ImageRestorer` | Shadow & vignetting removal, white balance, adaptive denoising, local contrast, mild sharpening, conservative deblurring |
| 4 | `content_preservation_guard` | Safety net: blends back toward the pre-restoration image if a step over-processed the page |
| 5 | `crop_to_document` | Adds a clean, configurable white margin |
| 6 | `ScannerSimulator` | Final brightness evening-out and edge clean-up so it reads as a scan, not a photo |

Non-rigid dewarping (Stage 2) and heavy restoration steps are skipped
automatically when they're not needed — e.g. a page detected as already
flat only gets a plain homography, per the "don't over-process" principle
that runs through the whole pipeline.

## Requirements

- Python 3.11+
- `opencv-contrib-python` — **not** plain `opencv-python`; the non-rigid
  dewarp step needs the contrib `createThinPlateSplineShapeTransformer`.
  (If only plain `opencv-python` is installed, the scanner still runs —
  it just logs a warning and falls back to perspective-only correction.)
- `numpy`
- `Pillow` — required for saving (DPI metadata + embedded sRGB profile)
- `scikit-image` — optional but recommended; enables the SSIM-based content
  guard (Stage 4) and a conservative deblurring step. Without it, both are
  silently skipped.

Install everything:

```bash
pip install opencv-contrib-python numpy Pillow scikit-image
```

GPU is used automatically for denoising when OpenCV reports a usable CUDA
device (`opencv-contrib-python` built with CUDA support), with a transparent
CPU fallback otherwise — no extra setup needed either way.

## Usage

```bash
python document_scanner.py <input_image> <output_image> [options]
```

Example:

```bash
python document_scanner.py IMG_1024.jpg cleaned_page.png
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--margin PX` | `15` | White margin, in pixels, added around the final page |
| `--dpi DPI` | `600` | DPI value embedded in the output PNG |
| `--device {auto,cpu,cuda}` | `auto` | Compute device for accelerated stages |
| `--strict` | off | Fail instead of falling back to the full frame when no document is confidently detected |
| `--force-homography` | off | Always use a plain perspective transform; skip curvature (TPS) correction |
| `--min-confidence FLOAT` | `0.35` | Detection confidence below which a louder warning is shown |
| `--debug` | off | Verbose logging and full tracebacks on error |
| `--version` | — | Print the version and exit |

### Supported input formats

JPG, JPEG, PNG, BMP, TIFF, WEBP. Output is always a lossless PNG.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Unexpected error |
| 2 | Invalid command-line arguments |
| 3 | Input file not found |
| 4 | Unsupported input format |
| 5 | Input image is corrupted / unreadable |
| 6 | No document detected (`--strict` only — otherwise it falls back to the full frame with a warning) |
| 7 | Failed to write the output file |
| 8 | An internal processing stage failed |

## As a library

```python
from pathlib import Path
from document_scanner import DocumentScannerPipeline, ScannerConfig

config = ScannerConfig(margin_px=20, force_homography=False)
pipeline = DocumentScannerPipeline(config)
result_image, report = pipeline.process(Path("IMG_1024.jpg"))

print(report.detection.confidence, report.applied_nonrigid_dewarp)
```

`result_image` is a BGR `numpy.ndarray`; use `save_image()` from the same
module to write it to disk with DPI/ICC metadata.

## Known limitations

- **Classical CV, not a learned model.** It handles curled/folded edges and
  book-gutter bulge well because those show up as boundary curvature, which
  `DocumentLocalizer` detects and `GeometricReconstructor` corrects. It will
  *not* recover interior text-line curvature when the page's outer
  silhouette is itself straight (e.g. a photographed open book where the
  spine hides the true boundary). A learned dewarping network could be
  dropped into `GeometricReconstructor` for that case without touching the
  rest of the pipeline.
- **No OCR-based orientation correction.** The output rectangle matches the
  page's true aspect ratio regardless of how it was rotated in the photo,
  but content orientation (e.g. a page shot sideways) is not auto-rotated
  upright — that would need text-direction detection, which is out of
  scope here.
- **Deblurring is intentionally conservative.** Only a light,
  bounded Richardson–Lucy pass runs, and only in a narrow "moderately
  blurred" band, to avoid the ringing artifacts that more aggressive/blind
  deconvolution tends to introduce.
