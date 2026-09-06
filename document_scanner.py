#!/usr/bin/env python3
"""High-quality AI document scanner.

Converts a photograph of a paper document into an image that closely
resembles a page produced by a professional flatbed scanner.

The pipeline is organised into six stages, each implemented as an
independent, unit-testable component so that any one of them (for example
the classical :class:`DocumentLocalizer`) can later be swapped for a
learned model without touching the rest of the code:

    1. :class:`DocumentLocalizer`      -- find the page in the photo
    2. :class:`GeometricReconstructor` -- flatten perspective + curvature
    3. :class:`ImageRestorer`          -- shadows, colour, noise, sharpness
    4. ``content_preservation_guard``  -- safety net against over-processing
    5. ``crop_to_document``            -- clean, margin-controlled cropping
    6. :class:`ScannerSimulator`       -- final "flatbed" look

Usage
-----
    python document_scanner.py <input_image> <output_image> [options]

Example
-------
    python document_scanner.py IMG_1024.jpg cleaned_page.png

Run ``python document_scanner.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

try:  # skimage is used for a couple of well-tested restoration primitives.
    from skimage.metrics import structural_similarity as _ssim
    from skimage.restoration import richardson_lucy as _richardson_lucy
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - skimage is an optional refinement.
    _HAVE_SKIMAGE = False

LOGGER = logging.getLogger("document_scanner")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Formats OpenCV/Pillow can read that we explicitly promise to support.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_OUTPUT_DPI = 600
MIN_RECOMMENDED_DIMENSION = 400  # px; below this we warn about resolution
DETECTION_WORKING_MAX_DIM = 1600  # downscale target for Stage 1 (speed/robustness)
MIN_DOCUMENT_AREA_RATIO = 0.10  # a candidate must cover >=10% of the frame
MAX_DOCUMENT_AREA_RATIO = 0.95  # candidates covering nearly the whole frame are
                                  # almost always an inverted-background artifact
FLATNESS_DEVIATION_RATIO = 0.006  # edge deviation / page-size => "curved" page
MAX_EDGE_SAMPLE_POINTS = 24  # points sampled per edge for the TPS solve
AMBIGUITY_SCORE_MARGIN = 0.10  # candidates within 10% of the best are "ambiguous"

# Process-wide exit codes (also used as unit tests for main()).
EXIT_OK = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_BAD_ARGS = 2
EXIT_FILE_NOT_FOUND = 3
EXIT_UNSUPPORTED_FORMAT = 4
EXIT_UNREADABLE_IMAGE = 5
EXIT_NO_DOCUMENT = 6
EXIT_OUTPUT_WRITE_FAILED = 7
EXIT_PROCESSING_FAILED = 8


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class DocumentScannerError(Exception):
    """Base class for all recoverable, user-facing scanner errors."""

    exit_code: int = EXIT_UNEXPECTED_ERROR


class InputFileNotFoundError(DocumentScannerError):
    exit_code = EXIT_FILE_NOT_FOUND


class UnsupportedFormatError(DocumentScannerError):
    exit_code = EXIT_UNSUPPORTED_FORMAT


class UnreadableImageError(DocumentScannerError):
    exit_code = EXIT_UNREADABLE_IMAGE


class DocumentNotFoundError(DocumentScannerError):
    exit_code = EXIT_NO_DOCUMENT


class OutputWriteError(DocumentScannerError):
    exit_code = EXIT_OUTPUT_WRITE_FAILED


class ProcessingError(DocumentScannerError):
    exit_code = EXIT_PROCESSING_FAILED


# --------------------------------------------------------------------------
# Configuration and result data classes
# --------------------------------------------------------------------------


@dataclass
class ScannerConfig:
    """User-tunable knobs for the pipeline.

    Attributes:
        margin_px: Uniform white margin (in pixels) added around the final
            cropped page. Sits at the boundary of Stage 5 (cropping) and
            Stage 6 (scanner simulation).
        output_dpi: DPI value embedded in the output PNG's metadata.
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"``. ``"auto"`` uses a
            GPU-accelerated code path when OpenCV reports a usable CUDA
            device, and transparently falls back to CPU otherwise.
        strict: If True, raise :class:`DocumentNotFoundError` when no page
            can be confidently located instead of falling back to treating
            the whole frame as the document.
        force_homography: If True, always skip the non-rigid (curvature)
            correction stage and use a plain 4-point perspective transform.
        min_confidence: Detections below this confidence are still used,
            but trigger a louder warning to the user.
        debug: Enables verbose logging and full tracebacks on error.
        max_working_dimension: Longest side (px) used internally during
            Stage 1 detection; the full-resolution image is always used for
            the actual geometric/photometric processing.
    """

    margin_px: int = 15
    output_dpi: int = DEFAULT_OUTPUT_DPI
    device: str = "auto"
    strict: bool = False
    force_homography: bool = False
    min_confidence: float = 0.35
    debug: bool = False
    max_working_dimension: int = DETECTION_WORKING_MAX_DIM


@dataclass
class DocumentDetection:
    """Result of Stage 1 (document localization), in *original* image scale."""

    corners: np.ndarray  # (4, 2) float32, ordered TL, TR, BR, BL
    confidence: float
    is_flat: bool
    used_fallback: bool
    ambiguous: bool
    edge_points: dict = field(default_factory=dict)  # {"top"/"right"/"bottom"/"left": (N,2)}


@dataclass
class PipelineReport:
    """Lightweight record of what happened, useful for --debug and logs."""

    detection: Optional[DocumentDetection] = None
    applied_nonrigid_dewarp: bool = False
    device_used: str = "cpu"
    warnings: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Small geometry helpers shared across stages
# --------------------------------------------------------------------------


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as (top-left, top-right, bottom-right, bottom-left).

    Args:
        pts: Array of shape (4, 2) in any order.

    Returns:
        Array of shape (4, 2), float32, in TL/TR/BR/BL order.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left has smallest x+y
    ordered[2] = pts[np.argmax(s)]  # bottom-right has largest x+y

    diff = np.diff(pts, axis=1).reshape(-1)  # x - y
    ordered[1] = pts[np.argmin(diff)]  # top-right has smallest x-y
    ordered[3] = pts[np.argmax(diff)]  # bottom-left has largest x-y
    return ordered


def polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula area for a small polygon given as an (N, 2) array."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def expand_quad(corners: np.ndarray, image_shape: tuple[int, int], fraction: float) -> np.ndarray:
    """Push each corner outward from the quad's centroid by ``fraction``.

    This gives Stage 5 (cropping) a small safety margin so that content
    right at the detected page edge is never clipped, while staying inside
    the bounds of the source image.

    Args:
        corners: (4, 2) ordered TL/TR/BR/BL corners.
        image_shape: (height, width) of the image the corners live in.
        fraction: Fraction of the corner-to-centroid distance to add.

    Returns:
        (4, 2) float32 array of expanded, clipped corners.
    """
    h, w = image_shape[:2]
    centroid = corners.mean(axis=0, keepdims=True)
    expanded = centroid + (corners - centroid) * (1.0 + fraction)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.float32)


def resolve_device(preference: str = "auto") -> str:
    """Decide whether to use a CUDA-accelerated OpenCV code path.

    Args:
        preference: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        ``"cuda"`` if CUDA should be used, else ``"cpu"``. Never raises --
        any inability to query CUDA support is treated as "not available".
    """
    if preference == "cpu":
        return "cpu"
    try:
        has_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:  # pragma: no cover - defensive; some builds lack cv2.cuda
        has_cuda = False
    if preference == "cuda" and not has_cuda:
        LOGGER.warning("CUDA was requested but no usable OpenCV CUDA device was found; using CPU.")
        return "cpu"
    return "cuda" if has_cuda else "cpu"


# --------------------------------------------------------------------------
# Stage 1 -- Document localization
# --------------------------------------------------------------------------


class DocumentLocalizer:
    """Finds a document's boundary, corners, and a rough flatness estimate.

    The detector combines two complementary candidate-generation strategies
    (gradient/edge based, and intensity/threshold based) because no single
    classical technique is robust across the full range of backgrounds and
    lighting described in the project brief. Candidates are scored and the
    best one wins; if nothing scores well enough the whole frame is used as
    a low-confidence fallback so the pipeline can always produce output.
    """

    def __init__(self, config: ScannerConfig) -> None:
        self.config = config

    # -- public API ---------------------------------------------------

    def locate(self, image: np.ndarray) -> DocumentDetection:
        """Locate the document in ``image`` (full-resolution, BGR uint8)."""
        h, w = image.shape[:2]
        scale = min(1.0, self.config.max_working_dimension / float(max(h, w)))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()

        candidates = self._gather_candidates(small)

        if not candidates:
            LOGGER.warning(
                "No confident document boundary found; falling back to the full frame. "
                "Results may include background."
            )
            corners_full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            return DocumentDetection(
                corners=corners_full,
                confidence=0.15,
                is_flat=True,
                used_fallback=True,
                ambiguous=False,
                edge_points={},
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        ambiguous = (
            len(candidates) > 1
            and (best["score"] - candidates[1]["score"]) < AMBIGUITY_SCORE_MARGIN * best["score"]
        )
        if ambiguous:
            LOGGER.warning(
                "Multiple plausible document regions were detected; the largest / "
                "best-scoring one was selected."
            )

        inv_scale = 1.0 / scale
        corners_full = (best["corners"] * inv_scale).astype(np.float32)

        is_flat = True
        edge_points_full: dict = {}
        if not best["used_fallback_shape"]:
            curvature_contour = self._extract_curvature_contour(small, best["corners"])
            if curvature_contour is not None:
                contour_full = curvature_contour.astype(np.float32) * inv_scale
                is_flat, edge_points_full = self._estimate_flatness(corners_full, contour_full)

        confidence = float(np.clip(best["score"], 0.0, 1.0))
        if confidence < self.config.min_confidence:
            LOGGER.warning(
                "Document detection confidence is low (%.2f); output quality may be reduced.",
                confidence,
            )

        return DocumentDetection(
            corners=order_points(corners_full),
            confidence=confidence,
            is_flat=is_flat,
            used_fallback=False,
            ambiguous=ambiguous,
            edge_points=edge_points_full,
        )

    # -- candidate generation ------------------------------------------

    def _gather_candidates(self, small: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        masks = []

        # Strategy A: gradient / edge based.
        median = float(np.median(smoothed))
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        edges = cv2.Canny(smoothed, lower, upper)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        masks.append(edges)

        # Strategy B: intensity / threshold based (handles low-texture, high
        # contrast page-vs-background scenes that Canny alone struggles with).
        _, otsu = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(otsu)
        masks.append(cv2.bitwise_not(otsu))

        frame_area = float(small.shape[0] * small.shape[1])
        candidates: list[dict] = []
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_DOCUMENT_AREA_RATIO * frame_area or area > MAX_DOCUMENT_AREA_RATIO * frame_area:
                    continue
                candidate = self._score_contour(contour, area, frame_area, small.shape[:2])
                if candidate is not None:
                    candidates.append(candidate)

        return self._deduplicate(candidates)

    def _score_contour(self, contour: np.ndarray, area: float, frame_area: float, shape: tuple[int, int]) -> Optional[dict]:
        # Light simplification to remove pixel-level jaggedness while
        # preserving genuine curvature (used later for flatness estimation).
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        smooth_contour = cv2.approxPolyDP(contour, epsilon=0.002 * perimeter, closed=True)

        hull = cv2.convexHull(smooth_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return None

        quad = cv2.approxPolyDP(hull, epsilon=0.02 * cv2.arcLength(hull, True), closed=True)
        used_fallback_shape = False
        if len(quad) == 4 and cv2.isContourConvex(quad):
            corners = quad.reshape(4, 2).astype(np.float32)
            shape_quality = area / hull_area  # how "quad-like" the raw region is
        else:
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect).astype(np.float32)
            shape_quality = 0.85  # discount: corners are a bounding approximation
            used_fallback_shape = True

        area_ratio = area / frame_area
        score = 0.6 * min(area_ratio, 1.0) + 0.4 * min(shape_quality, 1.0)

        return {
            "score": score,
            "corners": order_points(corners),
            "contour": smooth_contour,
            "used_fallback_shape": used_fallback_shape,
        }

    @staticmethod
    def _deduplicate(candidates: list[dict], iou_threshold: float = 0.85) -> list[dict]:
        """Collapse near-identical candidates from the different mask strategies."""
        kept: list[dict] = []
        for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
            is_dup = False
            for existing in kept:
                if DocumentLocalizer._quad_iou(cand["corners"], existing["corners"]) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
        return kept

    @staticmethod
    def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
        area_a = polygon_area(a)
        area_b = polygon_area(b)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        # Cheap approximate IoU via a shared raster mask (fine at this scale).
        x_min = int(min(a[:, 0].min(), b[:, 0].min()))
        y_min = int(min(a[:, 1].min(), b[:, 1].min()))
        x_max = int(max(a[:, 0].max(), b[:, 0].max())) + 1
        y_max = int(max(a[:, 1].max(), b[:, 1].max())) + 1
        w = max(x_max - x_min, 1)
        h = max(y_max - y_min, 1)
        mask_a = np.zeros((h, w), np.uint8)
        mask_b = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask_a, [(a - [x_min, y_min]).astype(np.int32)], 1)
        cv2.fillPoly(mask_b, [(b - [x_min, y_min]).astype(np.int32)], 1)
        inter = int(np.logical_and(mask_a, mask_b).sum())
        union = int(np.logical_or(mask_a, mask_b).sum())
        return inter / union if union else 0.0

    @staticmethod
    def _extract_curvature_contour(small: np.ndarray, quad_corners: np.ndarray) -> Optional[np.ndarray]:
        """Independently re-derive a high-fidelity boundary for curvature analysis.

        The quad used for the perspective transform may come from whichever
        mask scored best for *corner* localisation -- including a
        gradient/Canny based mask, whose dilate+close morphology is prone to
        bridging over concave, low-contrast notches (e.g. a curled or folded
        edge next to a dark background) and silently erasing exactly the
        curvature we need to detect. Plain intensity (Otsu) thresholding
        does not use that bridging morphology and is far more reliable at
        tracing such notches faithfully, so it is used here regardless of
        which mask won corner selection.

        Args:
            small: Working-resolution BGR image (same one passed to
                :meth:`_gather_candidates`).
            quad_corners: The winning candidate's (4, 2) corners, in the
                same coordinate space as ``small``.

        Returns:
            A lightly-simplified (N, 2) contour closely overlapping
            ``quad_corners``, or ``None`` if no sufficiently-overlapping
            intensity-based contour is found (curvature analysis is then
            skipped and the page is treated as flat).
        """
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        _, otsu = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_area = float(small.shape[0] * small.shape[1])

        best_contour = None
        best_iou = 0.0
        for mask in (otsu, cv2.bitwise_not(otsu)):
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_DOCUMENT_AREA_RATIO * frame_area or area > MAX_DOCUMENT_AREA_RATIO * frame_area:
                    continue
                points = contour.reshape(-1, 2).astype(np.float32)
                if len(points) < 4:
                    continue
                iou = DocumentLocalizer._quad_iou(points, quad_corners)
                if iou > best_iou:
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = max(0.001 * perimeter, 0.5)
                    simplified = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
                    best_iou = iou
                    best_contour = simplified.reshape(-1, 2).astype(np.float32)

        return best_contour if best_iou >= 0.6 else None

    # -- flatness / curvature estimation --------------------------------

    @staticmethod
    def _estimate_flatness(corners: np.ndarray, contour: np.ndarray) -> tuple[bool, dict]:
        """Check each edge of the quad for deviation from a straight line.

        Returns:
            ``(is_flat, edge_points)`` where ``edge_points`` maps
            ``"top"``/``"right"``/``"bottom"``/``"left"`` to the raw
            (sub-sampled) contour points that lie along that edge, in
            original-image coordinates. These are consumed by
            :class:`GeometricReconstructor` for non-rigid dewarping.
        """
        edge_names = ["top", "right", "bottom", "left"]
        corner_indices = [DocumentLocalizer._nearest_index(contour, c) for c in corners]

        is_flat = True
        edge_points: dict = {}
        edge_lengths = [
            float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)
        ]
        page_scale = max(np.mean(edge_lengths), 1.0)

        for i, name in enumerate(edge_names):
            start_idx = corner_indices[i]
            end_idx = corner_indices[(i + 1) % 4]
            segment = DocumentLocalizer._contour_slice(contour, start_idx, end_idx)
            if len(segment) < 3:
                edge_points[name] = np.stack([corners[i], corners[(i + 1) % 4]])
                continue

            p1, p2 = corners[i], corners[(i + 1) % 4]
            deviations = DocumentLocalizer._point_to_segment_distances(segment, p1, p2)
            max_dev = float(np.max(deviations)) if len(deviations) else 0.0
            if max_dev / page_scale > FLATNESS_DEVIATION_RATIO:
                is_flat = False

            edge_points[name] = DocumentLocalizer._resample_polyline(segment, MAX_EDGE_SAMPLE_POINTS)

        return is_flat, edge_points

    @staticmethod
    def _nearest_index(contour: np.ndarray, point: np.ndarray) -> int:
        dists = np.linalg.norm(contour - point.reshape(1, 2), axis=1)
        return int(np.argmin(dists))

    @staticmethod
    def _contour_slice(contour: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        if start_idx <= end_idx:
            forward = contour[start_idx : end_idx + 1]
            other = np.concatenate([contour[end_idx:], contour[: start_idx + 1]])
        else:
            forward = np.concatenate([contour[start_idx:], contour[: end_idx + 1]])
            other = contour[end_idx : start_idx + 1]

        # Pick whichever traversal is geometrically shorter (i.e. the actual
        # edge, not the rest of the page perimeter). Point *count* is not a
        # safe proxy here: a curved/folded edge can legitimately need more
        # vertices than the three straight edges combined, so we compare
        # physical arc length instead.
        def _arc_length(pts: np.ndarray) -> float:
            if len(pts) < 2:
                return 0.0
            return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

        return forward if _arc_length(forward) <= _arc_length(other) else other

    @staticmethod
    def _point_to_segment_distances(points: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec) + 1e-6
        line_unit = line_vec / line_len
        normal = np.array([-line_unit[1], line_unit[0]])
        return np.abs((points - p1.reshape(1, 2)) @ normal)

    @staticmethod
    def _resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
        if len(points) <= n:
            return points.astype(np.float32)
        idx = np.linspace(0, len(points) - 1, n).astype(int)
        return points[idx].astype(np.float32)


# --------------------------------------------------------------------------
# Stage 2 -- Geometric reconstruction (perspective + non-rigid dewarping)
# --------------------------------------------------------------------------


class GeometricReconstructor:
    """Flattens the document: a 4-point perspective transform, plus an
    optional thin-plate-spline (TPS) correction for residual curvature
    (curled edges, folds, mild book-gutter bulge) detected in Stage 1.

    A plain homography is used whenever the page was judged flat -- "if the
    page is already flat, unnecessary transformations should be avoided".
    """

    #: Minimum number of boundary correspondence points required before a
    #: TPS solve is attempted; below this the fit would be unreliable.
    MIN_TPS_POINTS = 8
    MARGIN_EXPANSION_FRACTION = 0.012

    def __init__(self, config: ScannerConfig) -> None:
        self.config = config

    def reconstruct(self, image: np.ndarray, detection: DocumentDetection) -> tuple[np.ndarray, bool]:
        """Return ``(flattened_image, applied_nonrigid_dewarp)``."""
        corners = order_points(detection.corners)
        expanded = order_points(expand_quad(corners, image.shape, self.MARGIN_EXPANSION_FRACTION))

        out_w, out_h = self._output_size(expanded)
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
        )
        homography = cv2.getPerspectiveTransform(expanded, dst)
        flattened = cv2.warpPerspective(
            image,
            homography,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        skip_dewarp = detection.is_flat or self.config.force_homography or not detection.edge_points
        if skip_dewarp:
            return flattened, False

        dewarped = self._nonrigid_dewarp(flattened, detection.edge_points, homography, out_w, out_h)
        if dewarped is None:
            return flattened, False
        return dewarped, True

    @staticmethod
    def _output_size(corners: np.ndarray) -> tuple[int, int]:
        (tl, tr, br, bl) = corners
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        return max(int(round(width)), 16), max(int(round(height)), 16)

    def _nonrigid_dewarp(
        self,
        flattened: np.ndarray,
        edge_points: dict,
        homography: np.ndarray,
        out_w: int,
        out_h: int,
    ) -> Optional[np.ndarray]:
        """Correct residual boundary curvature with a thin-plate spline.

        For each raw boundary point detected in Stage 1, we know (a) where
        it actually lands in the flattened image (by pushing it through the
        same homography) and (b) where it *should* land if the page edge
        were perfectly straight (its perpendicular projection onto the
        corresponding rectangle side). Feeding those correspondences to a
        TPS solver yields a smooth warp that pulls curled/folded edges
        straight while leaving already-straight regions essentially
        untouched.
        """
        ideal_points: list[list[float]] = []
        actual_points: list[list[float]] = []

        for name, pts in edge_points.items():
            if pts is None or len(pts) == 0:
                continue
            pts_h = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
            warped_pts = cv2.perspectiveTransform(pts_h, homography).reshape(-1, 2)
            for x, y in warped_pts:
                if name == "top":
                    ideal = (float(x), 0.0)
                elif name == "bottom":
                    ideal = (float(x), float(out_h - 1))
                elif name == "left":
                    ideal = (0.0, float(y))
                elif name == "right":
                    ideal = (float(out_w - 1), float(y))
                else:  # pragma: no cover - defensive, unknown edge name
                    continue
                actual_points.append([float(x), float(y)])
                ideal_points.append(list(ideal))

        if len(ideal_points) < self.MIN_TPS_POINTS:
            LOGGER.debug("Too few boundary correspondences for TPS dewarping; skipping.")
            return None

        # Adjacent edges each independently estimate the shared corner
        # between them, producing near-duplicate points with slightly
        # different coordinates. Thin-plate splines are ill-conditioned
        # when given several near-coincident, mutually inconsistent
        # constraints, which otherwise shows up as large-scale "fisheye"
        # ringing far from the actual curvature. Clustering and averaging
        # those near-duplicates keeps the correction localized.
        ideal_points, actual_points = self._merge_nearby_points(
            ideal_points, actual_points, radius=max(10.0, 0.015 * min(out_w, out_h))
        )
        if len(ideal_points) < self.MIN_TPS_POINTS:
            return None

        matches = [cv2.DMatch(i, i, 0) for i in range(len(ideal_points))]
        transforming_shape = np.array(ideal_points, dtype=np.float32).reshape(1, -1, 2)
        target_shape = np.array(actual_points, dtype=np.float32).reshape(1, -1, 2)

        try:
            tps = cv2.createThinPlateSplineShapeTransformer()
            # A small amount of regularization trades exact interpolation of
            # (possibly slightly noisy) boundary detections for a smoother,
            # numerically stable warp -- appropriate since we want the
            # *general* curvature corrected, not to chase pixel-level noise.
            tps.setRegularizationParameter(0.5)
            tps.estimateTransformation(transforming_shape, target_shape, matches)
            result = tps.warpImage(
                flattened,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        except cv2.error as exc:
            LOGGER.warning("Non-rigid dewarping failed (%s); using the perspective-only result.", exc)
            return None
        return result

    @staticmethod
    def _merge_nearby_points(
        ideal_points: list[list[float]], actual_points: list[list[float]], radius: float
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Cluster points whose *actual* positions are within ``radius`` px.

        Each cluster collapses to a single (ideal, actual) pair using the
        cluster's mean, which is what keeps shared corners from adjacent
        edges from fighting each other in the TPS solve.
        """
        actual_arr = np.asarray(actual_points, dtype=np.float32)
        ideal_arr = np.asarray(ideal_points, dtype=np.float32)
        n = len(actual_arr)
        used = np.zeros(n, dtype=bool)
        merged_ideal: list[list[float]] = []
        merged_actual: list[list[float]] = []

        for i in range(n):
            if used[i]:
                continue
            dists = np.linalg.norm(actual_arr - actual_arr[i], axis=1)
            cluster = np.where((dists < radius) & (~used))[0]
            used[cluster] = True
            merged_ideal.append(ideal_arr[cluster].mean(axis=0).tolist())
            merged_actual.append(actual_arr[cluster].mean(axis=0).tolist())

        return merged_ideal, merged_actual


# --------------------------------------------------------------------------
# Stage 3 -- Image restoration
# --------------------------------------------------------------------------


class ImageRestorer:
    """Shadow/illumination normalization, colour correction, denoising,
    contrast, and mild sharpening -- in that order, since each later step
    assumes the previous one has already made the image better-behaved.

    Every operation here is deliberately conservative: the brief calls for
    "no visible halos", "no ringing artifacts", "no over-sharpening", and
    for information preservation to take priority over cosmetic
    enhancement, so strengths are derived from measurements of the image
    itself (noise level, blur amount) rather than fixed, aggressive
    constants.
    """

    def __init__(self, config: ScannerConfig, device: str) -> None:
        self.config = config
        self.device = device

    def restore(self, image: np.ndarray) -> np.ndarray:
        working = image.astype(np.float32)

        working = self._normalize_illumination(working)
        working = self._white_balance(working)
        result = np.clip(working, 0, 255).astype(np.uint8)

        result = self._denoise(result)
        result = self._enhance_local_contrast(result)
        result = self._sharpen(result)
        result = self._conditional_deblur(result)
        result = self._whiten_background(result)
        return result

    # -- illumination / shadows / vignetting ----------------------------

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Divide out a smoothly-estimated background to remove shadows and
        vignetting in one step (both are slow, spatially-varying
        multiplicative attenuation, so the same correction handles both).
        """
        h, w = image.shape[:2]
        # Estimate the background at reduced resolution for speed and to
        # force the estimate to be genuinely low-frequency (page content
        # should not leak into it).
        small_w, small_h = max(w // 8, 32), max(h // 8, 32)
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

        kernel = max(3, (min(small_w, small_h) // 3) | 1)  # odd kernel
        background_small = cv2.medianBlur(small.astype(np.uint8), kernel).astype(np.float32)
        # A morphological close mops up any thin dark content the median
        # filter alone did not fully remove from the background estimate.
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        background_small = cv2.morphologyEx(background_small, cv2.MORPH_CLOSE, struct)
        background = cv2.resize(background_small, (w, h), interpolation=cv2.INTER_CUBIC)
        background = np.clip(background, 1.0, 255.0)

        normalized = image / background * 255.0
        return np.clip(normalized, 0, 255)

    # -- colour --------------------------------------------------------

    def _white_balance(self, image: np.ndarray, p: float = 6.0) -> np.ndarray:
        """"Shades of Gray" white balance: assumes the Minkowski p-norm of
        each channel should be roughly equal once colour casts (e.g. from
        colored indoor lighting) are removed.
        """
        channels = cv2.split(np.clip(image, 0, 255))
        norms = [float(np.power(np.mean(np.power(c.astype(np.float64), p)), 1.0 / p)) for c in channels]
        overall = float(np.mean(norms))
        gains = [overall / n if n > 1e-3 else 1.0 for n in norms]
        gains = [float(np.clip(g, 0.7, 1.4)) for g in gains]  # avoid extreme colour shifts
        balanced = cv2.merge([c * g for c, g in zip(channels, gains)])
        return np.clip(balanced, 0, 255)

    # -- denoising -------------------------------------------------------

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        sigma = self._estimate_noise_sigma(image)
        if sigma < 1.5:
            return image  # already clean; do not touch fine text detail

        strength = float(np.clip(sigma * 1.3, 3.0, 12.0))
        if self.device == "cuda":
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                gpu_out = cv2.cuda.fastNlMeansDenoisingColored(gpu_img, None, strength, strength, 7, 21)
                return gpu_out.download()
            except Exception as exc:  # pragma: no cover - depends on build
                LOGGER.debug("GPU denoising unavailable (%s); using CPU.", exc)
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

    @staticmethod
    def _estimate_noise_sigma(image: np.ndarray) -> float:
        """Robust per-image noise estimate via the median absolute deviation
        of a Laplacian response (standard, cheap, works well for photos).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        mad = float(np.median(np.abs(laplacian - np.median(laplacian))))
        return mad * 1.4826 / 6.0  # scale factor relating Laplacian MAD to pixel-noise sigma

    # -- contrast / sharpness --------------------------------------------

    def _enhance_local_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    def _sharpen(self, image: np.ndarray, amount: float = 0.6, sigma: float = 2.0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _conditional_deblur(self, image: np.ndarray) -> np.ndarray:
        """Conservative deconvolution, applied only in a narrow "moderately
        blurred" band. Heavier or blind deconvolution risks the ringing
        artifacts the project brief explicitly warns against, so this
        deliberately does *less* rather than risk that.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not (30.0 < blur_score < 120.0) or not _HAVE_SKIMAGE:
            return image

        psf = cv2.getGaussianKernel(5, 1.0)
        psf_2d = (psf @ psf.T).astype(np.float64)
        channels = cv2.split(image.astype(np.float64) / 255.0)
        restored = []
        for channel in channels:
            try:
                deconvolved = _richardson_lucy(channel, psf_2d, num_iter=3)
            except Exception:  # pragma: no cover - defensive
                return image
            restored.append(deconvolved)
        merged = cv2.merge(restored) * 255.0
        return np.clip(merged, 0, 255).astype(np.uint8)

    # -- background whitening --------------------------------------------

    def _whiten_background(self, image: np.ndarray) -> np.ndarray:
        """Gentle levels stretch anchored on the paper background, so the
        page ends up a clean, uniform white the way a flatbed scan would,
        without blowing out light (but non-white) annotations.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        background_level = float(np.percentile(gray, 90))
        if background_level < 5:
            return image
        gain = float(np.clip(250.0 / background_level, 1.0, 1.6))
        whitened = image.astype(np.float32) * gain
        return np.clip(whitened, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Stage 4 -- Content preservation guard
# --------------------------------------------------------------------------


def content_preservation_guard(
    pre_restoration: np.ndarray, post_restoration: np.ndarray, min_similarity: float = 0.55
) -> np.ndarray:
    """Safety net enforcing "information preservation over cosmetic gain".

    Restoration parameters in Stage 3 are chosen conservatively, but images
    vary enormously, so this acts as a last-resort guard: if the restored
    image has structurally diverged too far from the geometrically
    corrected (but not yet photometrically altered) image -- as measured by
    structural similarity -- it is blended back toward the original so that
    no single misbehaving step can destroy content.

    Args:
        pre_restoration: Image right after Stage 2 (geometry only).
        post_restoration: Image after Stage 3.
        min_similarity: SSIM threshold below which blending kicks in.

    Returns:
        ``post_restoration`` unchanged if it is structurally faithful,
        otherwise a blend that pulls it back toward ``pre_restoration``.
    """
    if not _HAVE_SKIMAGE:
        return post_restoration
    try:
        gray_pre = cv2.cvtColor(pre_restoration, cv2.COLOR_BGR2GRAY)
        gray_post = cv2.cvtColor(post_restoration, cv2.COLOR_BGR2GRAY)
        score = _ssim(gray_pre, gray_post)
    except Exception:  # pragma: no cover - defensive
        return post_restoration

    if score >= min_similarity:
        return post_restoration

    LOGGER.warning(
        "Restoration changed image structure more than expected (SSIM=%.2f); "
        "blending back toward the pre-restoration image to protect content.",
        score,
    )
    blend_weight = float(np.clip(score / min_similarity, 0.3, 1.0))
    blended = cv2.addWeighted(
        post_restoration, blend_weight, pre_restoration, 1 - blend_weight, 0
    )
    return blended


# --------------------------------------------------------------------------
# Stage 5 -- Cropping
# --------------------------------------------------------------------------


def crop_to_document(image: np.ndarray, margin_px: int) -> np.ndarray:
    """Add a small, uniform white margin around the (already tightly
    geometrically cropped, in Stage 2) page.

    The perspective/TPS warp in Stage 2 already produces an image whose
    frame *is* the detected page boundary (plus a tiny safety margin), so
    there is no further content to crop away here; this stage only adds
    the clean uniform border a flatbed scan typically has.
    """
    if margin_px <= 0:
        return image
    return cv2.copyMakeBorder(
        image, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


# --------------------------------------------------------------------------
# Stage 6 -- Scanner simulation
# --------------------------------------------------------------------------


class ScannerSimulator:
    """Final polish pass so the output reads as a flatbed scan rather than
    an obviously processed photo: a clean rectangle, uniform brightness,
    and no leftover processing artifacts at the very edge of the page.
    """

    def __init__(self, config: ScannerConfig) -> None:
        self.config = config

    def simulate(self, image: np.ndarray) -> np.ndarray:
        result = self._even_out_brightness(image)
        result = self._trim_edge_fringe(result)
        return result

    @staticmethod
    def _even_out_brightness(image: np.ndarray) -> np.ndarray:
        """A final, very mild global gamma nudge so the page brightness is
        consistent corner-to-corner. Intentionally subtle -- Stage 3 already
        did the heavy lifting; this just removes any last unevenness.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        if mean_val <= 1.0 or mean_val >= 254.0:
            return image
        target = 235.0
        gamma = float(np.clip(np.log(target / 255.0) / np.log(mean_val / 255.0), 0.85, 1.15))
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, lut)

    @staticmethod
    def _trim_edge_fringe(image: np.ndarray, fringe_px: int = 2) -> np.ndarray:
        """Removes a hairline of potentially-interpolated pixels right at
        the outer edge (an artifact of warping), replacing it by extending
        the clean interior outward. Keeps the final border crisp.
        """
        h, w = image.shape[:2]
        if h <= 4 * fringe_px or w <= 4 * fringe_px:
            return image
        inner = image[fringe_px : h - fringe_px, fringe_px : w - fringe_px]
        return cv2.copyMakeBorder(
            inner, fringe_px, fringe_px, fringe_px, fringe_px, cv2.BORDER_REPLICATE
        )


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk, raising a scanner-specific error on failure.

    Uses ``cv2.imdecode`` over raw bytes (rather than ``cv2.imread``) so
    that non-ASCII paths behave consistently across platforms.
    """
    if not path.exists():
        raise InputFileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise InputFileNotFoundError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported input format '{path.suffix}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise UnreadableImageError(f"Could not read '{path}': {exc}") from exc

    if image is None:
        raise UnreadableImageError(
            f"'{path}' could not be decoded as an image (it may be corrupted or empty)."
        )

    h, w = image.shape[:2]
    if min(h, w) < MIN_RECOMMENDED_DIMENSION:
        LOGGER.warning(
            "Input resolution is very low (%dx%d); output quality will be limited.", w, h
        )
    return image


def save_image(image: np.ndarray, path: Path, dpi: int) -> None:
    """Save ``image`` (BGR uint8) as a lossless PNG with DPI metadata and,
    when Pillow's ImageCms support is available, an embedded sRGB profile.
    """
    try:
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        save_kwargs = {"format": "PNG", "compress_level": 6, "dpi": (dpi, dpi)}
        try:
            from PIL import ImageCms

            srgb_profile = ImageCms.createProfile("sRGB")
            icc_bytes = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
            save_kwargs["icc_profile"] = icc_bytes
        except Exception:  # pragma: no cover - profile embedding is best-effort
            LOGGER.debug("Could not embed an sRGB profile; saving without one.")

        path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(str(path), **save_kwargs)
    except OutputWriteError:
        raise
    except Exception as exc:
        raise OutputWriteError(f"Failed to write output image to '{path}': {exc}") from exc


# --------------------------------------------------------------------------
# Pipeline orchestrator
# --------------------------------------------------------------------------


class DocumentScannerPipeline:
    """Wires the six stages together and produces a :class:`PipelineReport`
    alongside the final image, so callers (CLI or library users embedding
    this in a larger pipeline) can inspect what happened.
    """

    def __init__(self, config: Optional[ScannerConfig] = None) -> None:
        self.config = config or ScannerConfig()
        self.device = resolve_device(self.config.device)
        self.localizer = DocumentLocalizer(self.config)
        self.reconstructor = GeometricReconstructor(self.config)
        self.restorer = ImageRestorer(self.config, self.device)
        self.simulator = ScannerSimulator(self.config)

    def process(self, input_path: Path) -> tuple[np.ndarray, PipelineReport]:
        report = PipelineReport(device_used=self.device)
        image = load_image(input_path)

        try:
            detection = self.localizer.locate(image)
        except Exception as exc:
            raise ProcessingError(f"Document localization failed: {exc}") from exc
        report.detection = detection

        if detection.used_fallback and self.config.strict:
            raise DocumentNotFoundError(
                "No document boundary could be confidently detected in the input image."
            )
        if detection.ambiguous:
            report.warnings.append("Multiple plausible document regions were detected.")
        if detection.confidence < self.config.min_confidence:
            report.warnings.append(f"Low detection confidence: {detection.confidence:.2f}")

        try:
            geometric, applied_dewarp = self.reconstructor.reconstruct(image, detection)
        except Exception as exc:
            raise ProcessingError(f"Geometric reconstruction failed: {exc}") from exc
        report.applied_nonrigid_dewarp = applied_dewarp

        try:
            restored = self.restorer.restore(geometric)
            restored = content_preservation_guard(geometric, restored)
        except Exception as exc:
            raise ProcessingError(f"Image restoration failed: {exc}") from exc

        try:
            cropped = crop_to_document(restored, self.config.margin_px)
            final = self.simulator.simulate(cropped)
        except Exception as exc:
            raise ProcessingError(f"Final scanner-simulation pass failed: {exc}") from exc

        return final, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document_scanner.py",
        description=(
            "Convert a photograph of a paper document into an image that "
            "closely resembles a professional flatbed scan."
        ),
    )
    parser.add_argument("input_image", type=str, help="Path to the input photograph.")
    parser.add_argument("output_image", type=str, help="Path to write the resulting PNG to.")
    parser.add_argument(
        "--margin",
        type=int,
        default=ScannerConfig.margin_px,
        metavar="PX",
        help=f"White margin, in pixels, added around the final page (default: {ScannerConfig.margin_px}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=ScannerConfig.output_dpi,
        help=f"DPI value embedded in the output PNG (default: {ScannerConfig.output_dpi}).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=ScannerConfig.device,
        help="Compute device for accelerated stages (default: auto).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of falling back to the full frame when no document is confidently detected.",
    )
    parser.add_argument(
        "--force-homography",
        action="store_true",
        help="Always use a plain 4-point perspective transform; skip curvature (TPS) correction.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=ScannerConfig.min_confidence,
        help="Detection confidence below which a louder warning is shown (default: %(default)s).",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging and full tracebacks.")
    parser.add_argument(
        "--version", action="version", version="High-Quality AI Document Scanner 1.0.0"
    )
    return parser


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.debug)

    config = ScannerConfig(
        margin_px=max(0, args.margin),
        output_dpi=max(1, args.dpi),
        device=args.device,
        strict=args.strict,
        force_homography=args.force_homography,
        min_confidence=args.min_confidence,
        debug=args.debug,
    )

    input_path = Path(args.input_image)
    output_path = Path(args.output_image)

    start = time.time()
    try:
        pipeline = DocumentScannerPipeline(config)
        result, report = pipeline.process(input_path)
        save_image(result, output_path, dpi=config.output_dpi)
    except DocumentScannerError as exc:
        LOGGER.error(str(exc))
        if args.debug:
            traceback.print_exc()
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - final safety net
        LOGGER.error("An unexpected error occurred: %s", exc)
        if args.debug:
            traceback.print_exc()
        return EXIT_UNEXPECTED_ERROR

    elapsed = time.time() - start
    LOGGER.info(
        "Saved '%s' (%dx%d, %s) in %.2fs [device=%s, dewarp=%s, confidence=%.2f]",
        output_path,
        result.shape[1],
        result.shape[0],
        f"{config.output_dpi} DPI",
        elapsed,
        report.device_used,
        report.applied_nonrigid_dewarp,
        report.detection.confidence if report.detection else 0.0,
    )
    for warning in report.warnings:
        LOGGER.warning(warning)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
