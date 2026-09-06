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

try:
    from scipy.interpolate import PchipInterpolator as _PchipInterpolator
    _HAVE_SCIPY_INTERPOLATION = True
except Exception:  # pragma: no cover - optional non-linear dewarp refinement
    _HAVE_SCIPY_INTERPOLATION = False

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
MAX_EDGE_SAMPLE_POINTS = 48  # boundary samples retained for spline/mesh dewarping
DEWARP_GRID_X = 48  # horizontal mesh cells
DEWARP_GRID_Y = 48  # vertical mesh cells
AMBIGUITY_SCORE_MARGIN = 0.10
GRABCUT_BORDER_RATIO = 0.025  # outer band seeded as definite background
GRABCUT_MIN_ITERATIONS = 5
GRABCUT_CLOSE_KERNEL = 31  # candidates within 10% of the best are "ambiguous"

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

        # Derive curvature samples from the actual outer contour rather than
        # from arbitrary high-gradient features inside the document. Edge maps
        # (Canny) are already used during candidate generation; here we use the
        # resulting page contour so the spline stage cannot accidentally lock
        # onto text strokes, dimension lines, or background texture.
        curvature_contour = self._extract_curvature_contour(small, best["corners"])
        if curvature_contour is None:
            candidate_contour = best.get("contour")
            curvature_contour = (
                np.asarray(candidate_contour, dtype=np.float32).reshape(-1, 2)
                if candidate_contour is not None else None
            )
        if curvature_contour is not None:
            refined_small_edges = self._extract_edge_points_from_contour(
                curvature_contour, best["corners"]
            )
            if self._edge_points_usable(refined_small_edges):
                edge_points_full = {
                    k: np.asarray(v, dtype=np.float32) * inv_scale
                    for k, v in refined_small_edges.items()
                }
                is_flat = self._estimate_flatness_from_edges(
                    best["corners"], refined_small_edges
                )

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

        candidates = self._deduplicate(candidates)

        # Once a genuine quadrilateral exists, discard full-frame fallback
        # rectangles before ambiguity scoring.  Otherwise the fallback can
        # appear "close enough" in score simply because it has a huge area.
        non_fallback = [c for c in candidates if not c.get("used_fallback_shape", False)]
        if non_fallback:
            candidates = non_fallback

        # A common failure mode is a huge minAreaRect fallback that spans the
        # entire photograph.  That candidate is attractive because it scores
        # well on area, but it is clearly not a page when it is clipped by the
        # image border on several sides.  In that situation run a second,
        # colour-aware foreground segmentation pass before giving up.
        if self._needs_foreground_segmentation(candidates, small.shape[:2]):
            grabcut_candidate = self._grabcut_candidate(small)
            if grabcut_candidate is not None:
                candidates.append(grabcut_candidate)
                candidates = self._deduplicate(candidates)

        return candidates

    @staticmethod
    def _needs_foreground_segmentation(candidates: list[dict], shape: tuple[int, int]) -> bool:
        """Return True when classical contour proposals are likely background.

        In particular, a min-area-rectangle fallback that touches multiple
        image borders is usually the photographic background rather than the
        sheet of paper.
        """
        if not candidates:
            return True
        h, w = shape
        border_x = max(4.0, 0.02 * w)
        border_y = max(4.0, 0.02 * h)
        best = max(candidates, key=lambda c: c["score"])
        corners = best["corners"]
        touches = int(np.sum(corners[:, 0] <= border_x))
        touches += int(np.sum(corners[:, 0] >= w - 1 - border_x))
        touches += int(np.sum(corners[:, 1] <= border_y))
        touches += int(np.sum(corners[:, 1] >= h - 1 - border_y))
        return bool(best.get("used_fallback_shape", False) and touches >= 2)

    def _grabcut_candidate(self, small: np.ndarray) -> Optional[dict]:
        """Use border-seeded GrabCut to isolate a paper sheet from the photo.

        GrabCut is deliberately used as a *fallback*, not the primary detector:
        it is strongest on photographs where the page is large and visually
        distinct from the surrounding table/fabric/background, exactly the
        situation in which contour-based thresholding can accidentally return
        the whole frame.
        """
        h, w = small.shape[:2]
        if min(h, w) < 64:
            return None

        try:
            mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
            bx = max(2, int(round(GRABCUT_BORDER_RATIO * w)))
            by = max(2, int(round(GRABCUT_BORDER_RATIO * h)))
            mask[:by, :] = cv2.GC_BGD
            mask[h - by :, :] = cv2.GC_BGD
            mask[:, :bx] = cv2.GC_BGD
            mask[:, w - bx :] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                small,
                mask,
                None,
                bgd_model,
                fgd_model,
                GRABCUT_MIN_ITERATIONS,
                cv2.GC_INIT_WITH_MASK,
            )

            foreground = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)

            close_kernel = max(5, GRABCUT_CLOSE_KERNEL | 1)
            structure = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
            )
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_CLOSE, structure, iterations=1
            )
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )

            contours, _ = cv2.findContours(
                foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            frame_area = float(h * w)
            valid = [
                c for c in contours
                if MIN_DOCUMENT_AREA_RATIO * frame_area
                <= cv2.contourArea(c)
                <= MAX_DOCUMENT_AREA_RATIO * frame_area
            ]
            if not valid:
                return None

            contour = max(valid, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area <= 0:
                return None

            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if perimeter <= 0 or hull_area <= 0:
                return None

            # Use a relatively tight approximation first; the convex hull
            # prevents small segmentation holes from creating fake corners.
            quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                corners = quad.reshape(4, 2).astype(np.float32)
                shape_quality = min(area / hull_area, 1.0)
            else:
                rect = cv2.minAreaRect(hull)
                corners = cv2.boxPoints(rect).astype(np.float32)
                shape_quality = 0.88

            # Reject pathological outputs that are effectively the frame.
            corners = order_points(corners)
            frame_margin = 0.01 * min(h, w)
            touches = np.sum(
                (corners[:, 0] <= frame_margin)
                | (corners[:, 0] >= w - 1 - frame_margin)
                | (corners[:, 1] <= frame_margin)
                | (corners[:, 1] >= h - 1 - frame_margin)
            )
            if touches >= 3:
                return None

            area_ratio = area / frame_area
            score = 0.60 * min(area_ratio, 1.0) + 0.40 * min(shape_quality, 1.0)
            return {
                "score": float(score),
                "corners": corners,
                "contour": contour,
                "used_fallback_shape": False,
            }
        except Exception as exc:
            LOGGER.debug("GrabCut foreground segmentation failed (%s); ignoring it.", exc)
            return None

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
    def _edge_points_usable(edge_points: dict) -> bool:
        """Return True only when every page side has good contour coverage."""
        if not edge_points:
            return False
        return all(
            name in edge_points and len(np.asarray(edge_points[name])) >= 8
            for name in ("top", "right", "bottom", "left")
        )

    @staticmethod
    def _extract_edge_points_from_contour(contour: np.ndarray, quad_corners: np.ndarray) -> dict:
        """Partition an outer page contour into four ordered edge point sets.

        Each contour point is assigned to an edge by its projection onto the
        corresponding corner-to-corner segment. A generous normal band permits
        bowed/curled edges to remain represented, while the requirement that
        each side spans most of its expected length rejects partial or unrelated
        contours.
        """
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 16:
            return {}
        corners = order_points(quad_corners)
        side_lengths = [
            float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
            for i in range(4)
        ]
        page_scale = max(min(side_lengths), 1.0)
        band = max(6.0, 0.10 * page_scale)
        result: dict = {}

        for i, name in enumerate(("top", "right", "bottom", "left")):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            d = p1 - p0
            d2 = float(np.dot(d, d))
            if d2 <= 1e-6:
                return {}
            t = ((pts - p0) @ d) / d2
            projected = p0 + t[:, None] * d
            distance = np.linalg.norm(pts - projected, axis=1)
            keep = (t >= -0.02) & (t <= 1.02) & (distance <= band)
            if int(np.count_nonzero(keep)) < 8:
                return {}

            selected_t = t[keep]
            coverage = float(selected_t.max() - selected_t.min())
            if coverage < 0.65:
                return {}

            selected = pts[keep]
            selected = selected[np.argsort(selected_t)]
            # Keep endpoint ownership deterministic. The spline fitting stage
            # will pin these to the exact detected corners.
            result[name] = DocumentLocalizer._resample_polyline(selected, MAX_EDGE_SAMPLE_POINTS)

        return result

    @staticmethod
    def _estimate_flatness_from_edges(corners: np.ndarray, edge_points: dict) -> bool:
        """Estimate curvature by measuring edge deviation from its chord."""
        corners = order_points(corners)
        lengths = [
            float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
            for i in range(4)
        ]
        page_scale = max(float(np.mean(lengths)), 1.0)
        worst_ratio = 0.0
        for i, name in enumerate(("top", "right", "bottom", "left")):
            pts = np.asarray(edge_points.get(name, []), dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            deviation = DocumentLocalizer._point_to_segment_distances(
                pts, corners[i], corners[(i + 1) % 4]
            )
            if len(deviation):
                worst_ratio = max(worst_ratio, float(np.max(deviation)) / page_scale)
        return worst_ratio <= FLATNESS_DEVIATION_RATIO

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
    """Flatten a perspective-distorted and gently curved document.

    A flat page is reconstructed with an ordinary four-point homography.
    When Stage 1 reports meaningful edge curvature, this stage instead builds
    a *non-linear document coordinate map* from four fitted cubic splines:

      1. each detected page edge is parameterised by normalized arc length;
      2. opposite edges are sampled at identical parameter values;
      3. a Coons-style surface interpolates those four curved boundaries into
         a dense source-coordinate mesh;
      4. the corresponding destination mesh is a uniform rectangle, so the
         curved source quadrilaterals are mapped to regular rectangular cells;
      5. ``cv2.remap`` performs the final inverse spatial warp.

    This is a mesh-warp formulation of non-linear dewarping. It avoids the
    optional OpenCV TPS shape-transformer API, which is not present in many
    standard OpenCV Python wheels, while still producing a genuinely
    non-rigid warp.
    """

    MARGIN_EXPANSION_FRACTION = 0.012
    MIN_SPLINE_POINTS = 4

    def __init__(self, config: ScannerConfig) -> None:
        self.config = config

    def reconstruct(self, image: np.ndarray, detection: DocumentDetection) -> tuple[np.ndarray, bool]:
        """Return ``(flattened_image, applied_nonrigid_dewarp)``.

        Perspective is corrected first. If boundary samples are available,
        they are transformed into that perspective-corrected coordinate system,
        where a spline/mesh warp removes the *residual non-linear curvature*.
        Thus a genuinely flat page is essentially left alone by the mesh stage,
        while a bowed/curled page is pulled onto a rectangular document grid.
        """
        corners = order_points(detection.corners)
        expanded = order_points(expand_quad(corners, image.shape, self.MARGIN_EXPANSION_FRACTION))

        flattened, homography, out_w, out_h = self._perspective_reconstruct(image, expanded)

        if not self.config.force_homography and self._edge_data_is_usable(detection.edge_points):
            dewarped = self._mesh_dewarp(
                flattened=flattened,
                edge_points=detection.edge_points,
                homography=homography,
                out_w=out_w,
                out_h=out_h,
            )
            if dewarped is not None:
                return dewarped, True
            LOGGER.warning("Non-linear mesh dewarping could not be constructed; using perspective-only reconstruction.")

        return flattened, False

    @staticmethod
    def _output_size(corners: np.ndarray) -> tuple[int, int]:
        (tl, tr, br, bl) = corners
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        return max(int(round(width)), 16), max(int(round(height)), 16)

    @classmethod
    def _perspective_reconstruct(cls, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        out_w, out_h = cls._output_size(corners)
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
        )
        homography = cv2.getPerspectiveTransform(corners, dst)
        flattened = cv2.warpPerspective(
            image,
            homography,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return flattened, homography, out_w, out_h

    @classmethod
    def _edge_data_is_usable(cls, edge_points: dict) -> bool:
        if not edge_points:
            return False
        names = ("top", "right", "bottom", "left")
        return all(name in edge_points and len(np.asarray(edge_points[name])) >= cls.MIN_SPLINE_POINTS for name in names)

    @staticmethod
    def _transform_edge_points(edge_points: dict, homography: np.ndarray) -> dict:
        """Transform all detected edge points into the perspective-corrected frame."""
        transformed: dict = {}
        for name, points in edge_points.items():
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            if len(pts) < 2:
                transformed[name] = pts.reshape(-1, 2)
                continue
            transformed[name] = cv2.perspectiveTransform(pts, homography).reshape(-1, 2).astype(np.float32)
        return transformed

    @staticmethod
    def _fit_edge_spline(points: np.ndarray, start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
        """Fit a shape-preserving cubic spline and sample it by normalized arc length."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2:
            return np.linspace(start, end, n, dtype=np.float32)

        # Remove consecutive duplicates; they make spline parameterisation singular.
        deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.r_[True, deltas > 1e-3]
        pts = pts[keep]

        # Make sure the contour runs in the intended corner-to-corner direction.
        start_d = np.linalg.norm(pts[0] - start)
        start_d_end = np.linalg.norm(pts[-1] - start)
        if start_d_end < start_d:
            pts = pts[::-1]

        # Endpoints are pinned exactly to the desired corners.
        pts[0] = np.asarray(start, dtype=np.float32)
        pts[-1] = np.asarray(end, dtype=np.float32)

        if len(pts) < 4 or not _HAVE_SCIPY_INTERPOLATION:
            t = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
            tq = np.linspace(0.0, 1.0, n, dtype=np.float32)
            x = np.interp(tq, t, pts[:, 0])
            y = np.interp(tq, t, pts[:, 1])
            return np.column_stack([x, y]).astype(np.float32)

        # Normalized physical arc length makes point correspondence meaningful.
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts.astype(np.float64), axis=0), axis=1))]
        total = float(arc[-1])
        if total <= 1e-6:
            return np.linspace(start, end, n, dtype=np.float32)
        t = (arc / total).astype(np.float64)
        tq = np.linspace(0.0, 1.0, n, dtype=np.float64)

        try:
            sx = _PchipInterpolator(t, pts[:, 0].astype(np.float64))(tq)
            sy = _PchipInterpolator(t, pts[:, 1].astype(np.float64))(tq)
            sampled = np.column_stack([sx, sy]).astype(np.float32)
        except Exception:
            x = np.interp(tq, t, pts[:, 0])
            y = np.interp(tq, t, pts[:, 1])
            sampled = np.column_stack([x, y]).astype(np.float32)

        sampled[0] = np.asarray(start, dtype=np.float32)
        sampled[-1] = np.asarray(end, dtype=np.float32)
        return sampled

    @staticmethod
    def _coons_patch(top: np.ndarray, right: np.ndarray, bottom: np.ndarray, left: np.ndarray,
                     nu: int, nv: int) -> np.ndarray:
        """Build a source-coordinate Coons patch from four compatible edge splines.

        ``top``/``bottom`` each contain ``nu`` points and ``left``/``right`` each
        contain ``nv`` points. The result has shape ``(nv, nu, 2)``.
        """
        u = np.linspace(0.0, 1.0, nu, dtype=np.float32)[None, :, None]
        v = np.linspace(0.0, 1.0, nv, dtype=np.float32)[:, None, None]

        top2 = top[None, :, :]
        bottom2 = bottom[None, :, :]
        left2 = left[:, None, :]
        right2 = right[:, None, :]

        c00 = top[0]
        c10 = top[-1]
        c01 = bottom[0]
        c11 = bottom[-1]
        bilinear = (
            (1.0 - u) * (1.0 - v) * c00
            + u * (1.0 - v) * c10
            + (1.0 - u) * v * c01
            + u * v * c11
        )
        return ((1.0 - v) * top2 + v * bottom2 + (1.0 - u) * left2 + u * right2 - bilinear).astype(np.float32)

    @staticmethod
    def _mesh_remap_from_nodes(source_nodes: np.ndarray, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert source mesh nodes into dense destination->source remap fields."""
        nv, nu, _ = source_nodes.shape
        if nu < 2 or nv < 2:
            raise ValueError("Mesh must contain at least two nodes in each direction")

        gx = np.linspace(0.0, nu - 1.0, out_w, dtype=np.float32)
        gy = np.linspace(0.0, nv - 1.0, out_h, dtype=np.float32)
        ix = np.minimum(np.floor(gx).astype(np.int32), nu - 2)
        iy = np.minimum(np.floor(gy).astype(np.int32), nv - 2)
        tx = gx - ix
        ty = gy - iy

        map_x = np.empty((out_h, out_w), dtype=np.float32)
        map_y = np.empty((out_h, out_w), dtype=np.float32)

        for row in range(out_h):
            j = int(iy[row])
            fy = float(ty[row])
            p00 = source_nodes[j, ix]
            p10 = source_nodes[j, ix + 1]
            p01 = source_nodes[j + 1, ix]
            p11 = source_nodes[j + 1, ix + 1]
            top = p00 + (p10 - p00) * tx[:, None]
            bottom = p01 + (p11 - p01) * tx[:, None]
            p = top + (bottom - top) * fy
            map_x[row] = p[:, 0]
            map_y[row] = p[:, 1]

        return map_x, map_y

    def _mesh_dewarp(
        self,
        flattened: np.ndarray,
        edge_points: dict,
        homography: np.ndarray,
        out_w: int,
        out_h: int,
    ) -> Optional[np.ndarray]:
        """Flatten residual page curvature with a spline-driven quadrilateral mesh.

        The page has already undergone perspective correction. The detected
        boundary samples are transformed into this flat coordinate system, fit
        with four splines, and sampled into matched top/bottom and left/right
        point sets. The four splines then define a curvilinear source mesh, while
        the destination mesh is a uniform rectangle. ``cv2.remap`` maps every
        destination pixel through that mesh.
        """
        transformed = self._transform_edge_points(edge_points, homography)
        tl = np.array([0.0, 0.0], dtype=np.float32)
        tr = np.array([out_w - 1.0, 0.0], dtype=np.float32)
        br = np.array([out_w - 1.0, out_h - 1.0], dtype=np.float32)
        bl = np.array([0.0, out_h - 1.0], dtype=np.float32)

        nu = DEWARP_GRID_X + 1
        nv = DEWARP_GRID_Y + 1
        top = self._fit_edge_spline(transformed["top"], tl, tr, nu)
        right = self._fit_edge_spline(transformed["right"], tr, br, nv)
        bottom = self._fit_edge_spline(transformed["bottom"], br, bl, nu)
        left = self._fit_edge_spline(transformed["left"], bl, tl, nv)

        # Explicitly enforce common corners so independent splines cannot fight.
        top[0], top[-1] = tl, tr
        right[0], right[-1] = tr, br
        bottom[0], bottom[-1] = br, bl
        left[0], left[-1] = bl, tl

        source_nodes = self._coons_patch(top, right, bottom, left, nu, nv)
        map_x, map_y = self._mesh_remap_from_nodes(source_nodes, out_w, out_h)

        # The mesh lives in the already-warped image, so valid coordinates are
        # simply the flattened image bounds.
        fh, fw = flattened.shape[:2]
        map_x = np.clip(map_x, 0, fw - 1)
        map_y = np.clip(map_y, 0, fh - 1)

        try:
            return cv2.remap(
                flattened,
                map_x,
                map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error as exc:
            LOGGER.warning("Mesh remapping failed: %s", exc)
            return None


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
