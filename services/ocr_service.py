"""
OCR Service - Extracts text from sticky note images using OpenCV and PaddleOCR.
HSV color segmentation, preprocessing, spatial linking, and background processing.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Initialize PaddleOCR (runs on CPU by default, no external binaries needed)
# use_textline_orientation=True enables text orientation detection
# lang='en' for English, can be changed to other languages
try:
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
    logger.info("PaddleOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {e}")
    ocr_engine = None


# ============================================================================
# Color Detection Configuration
# ============================================================================

# HSV color ranges for sticky note colors
# Format: (lower_bound, upper_bound) in HSV color space
# VERY PERMISSIVE ranges to catch all pastel/bright sticky notes
OCR_COLOR_RANGES = {
    "pink": (
        np.array([150, 15, 100]),
        np.array([180, 255, 255]),
    ),  # Pink/magenta sticky notes
    "red": (
        np.array([0, 15, 100]),
        np.array([10, 255, 255]),
    ),  # Red sticky notes
    "orange": (
        np.array([10, 15, 100]),
        np.array([25, 255, 255]),
    ),  # Orange sticky notes
    "yellow": (
        np.array([25, 15, 100]),
        np.array([35, 255, 255]),
    ),  # Yellow sticky notes
    "lime": (
        np.array([35, 15, 100]),
        np.array([75, 255, 255]),
    ),  # Light green/lime sticky notes
    "green": (
        np.array([75, 15, 100]),
        np.array([90, 255, 255]),
    ),  # Green sticky notes
    "cyan": (
        np.array([90, 15, 100]),
        np.array([100, 255, 255]),
    ),  # Cyan/turquoise sticky notes
    "blue": (
        np.array([100, 15, 100]),
        np.array([130, 255, 255]),
    ),  # Blue sticky notes
    "purple": (
        np.array([130, 15, 100]),
        np.array([150, 255, 255]),
    ),  # Purple/violet sticky notes
    "white": (
        np.array([0, 0, 180]),
        np.array([180, 30, 255]),
    ),  # White/very light sticky notes
    "gray": (
        np.array([0, 0, 100]),
        np.array([180, 30, 180]),
    ),  # Light gray sticky notes
}

# Default processing parameters
DEFAULT_HSV_TOLERANCE = 20  # Standard tolerance
DEFAULT_MIN_SIZE = 1500  # Sticky notes area threshold (e.g., 40x40 = 1600)
DEFAULT_MAX_SIZE = 50000
DEFAULT_PROXIMITY = 100


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class StickyRegion:
    """Represents a detected sticky note region."""

    id: int
    color_hex: str
    color_name: str
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    linked_to: List[int]  # IDs of spatially linked regions


# ============================================================================
# OCR Processing Class
# ============================================================================


class OCRService:
    """Handles OCR processing of sticky note images."""

    def __init__(
        self,
        hsv_tolerance: int = DEFAULT_HSV_TOLERANCE,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        proximity: int = DEFAULT_PROXIMITY,
    ):
        """
        Initialize OCR service.

        Args:
            hsv_tolerance: HSV color matching tolerance
            min_size: Minimum region size in pixels
            max_size: Maximum region size in pixels
            proximity: Proximity threshold for spatial linking in pixels
        """
        self.hsv_tolerance = hsv_tolerance
        self.min_size = min_size
        self.max_size = max_size
        self.proximity = proximity

    def process_image(
        self, image_path: str, progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process sticky note image and extract text regions.

        Args:
            image_path: Path to input image
            progress_callback: Callback(current, total, status, message, preview_url)

        Returns:
            List of region dictionaries
        """
        logger.info(f"Starting OCR processing: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # ROBUST APPROACH:
        # 1. Detect sticky note boundaries using color segmentation
        # 2. For each sticky note, extract text using PaddleOCR
        # This prevents mixing text from different sticky notes

        if ocr_engine is None:
            raise RuntimeError("PaddleOCR not initialized")

        # Convert to HSV for color-based sticky note detection with smoothing
        hsv = self._prepare_hsv_for_detection(image)

        # Run a single global text detection pass for alignment/validation
        global_text_boxes = self._run_global_text_detection(image)

        # Detect sticky note regions using color segmentation first
        sticky_note_regions = self._detect_color_regions(image, hsv)
        logger.info(
            f"Detected {len(sticky_note_regions)} sticky note regions via color segmentation"
        )

        # Run Paddle-based fallback when color detection looks weak
        fallback_regions: List[Dict[str, Any]] = []
        if self._should_use_paddle_fallback(sticky_note_regions, image.shape):
            fallback_regions = self._detect_regions_with_paddle(
                image, global_text_boxes
            )
            if fallback_regions:
                sticky_note_regions.extend(fallback_regions)
                logger.info(
                    f"Fallback detection added {len(fallback_regions)} PaddleOCR-derived regions"
                )
            else:
                logger.info("Fallback detection ran but found no additional regions")

        adaptive_regions = self._detect_adaptive_color_regions(
            image, hsv, sticky_note_regions
        )
        if adaptive_regions:
            sticky_note_regions.extend(adaptive_regions)
            logger.info(
                f"Adaptive color clustering added {len(adaptive_regions)} supplemental regions"
            )

        sticky_note_regions = self._validate_regions_with_text(
            sticky_note_regions, global_text_boxes, image.shape
        )
        sticky_note_regions = self._merge_candidate_regions(
            sticky_note_regions, iou_threshold=0.65, distance_threshold=35.0
        )

        if not sticky_note_regions:
            logger.warning("No sticky note candidates detected after validation")
            return []

        # Now extract text from each sticky note region
        regions = []
        region_id = 0
        total_regions = len(sticky_note_regions)

        for sticky in sticky_note_regions:
            x, y, w, h = sticky["bbox"]

            # Extract ROI
            roi = image[y : y + h, x : x + w]

            # Get dominant color
            color_hex = self._get_dominant_color(roi)

            pre_text = sticky.get("pre_text")
            combined_text = pre_text.strip() if isinstance(pre_text, str) else ""
            avg_conf = self._safe_float(sticky.get("pre_confidence"))

            if not combined_text:
                composed_text, composed_conf = self._compose_text_from_boxes(
                    (x, y, w, h), global_text_boxes
                )
                if composed_text:
                    combined_text = composed_text
                    avg_conf = composed_conf

            if not combined_text:
                combined_text, avg_conf = self._extract_text(roi)

            if not combined_text:
                logger.debug(
                    f"No text in {sticky.get('color_name', 'unknown')} sticky at ({x},{y},{w}x{h})"
                )
                continue

            logger.info(
                f"Region {region_id} ({sticky['color_name']}): '{combined_text[:60]}...' conf={avg_conf:.1f}%"
            )

            # Create region
            region = StickyRegion(
                id=region_id,
                color_hex=color_hex,
                color_name=sticky["color_name"],
                text=combined_text,
                bbox=(x, y, w, h),
                confidence=avg_conf,
                linked_to=[],
            )
            regions.append(region)
            region_id += 1

            # Report progress
            if progress_callback:
                progress_callback(
                    current=region_id,
                    total=total_regions,
                    status="processing",
                    message=f"Extracted {sticky['color_name']} sticky note",
                )

        logger.info(f"Detection complete: found {len(regions)} sticky notes")

        # Remove duplicate/overlapping regions
        regions = self._remove_duplicates(regions)
        logger.info(f"After deduplication: {len(regions)} unique regions")

        # Consolidate similar colors to prevent duplicate color mappings
        regions = self._consolidate_colors(regions)

        # Spatial linking (find description stickies near summary stickies)
        self._link_regions(regions)

        # Convert to dictionary format
        result = [self._region_to_dict(r) for r in regions]

        logger.info(f"OCR processing complete: {len(result)} regions extracted")
        return result

    def _detect_color_regions(
        self, image: np.ndarray, hsv: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect sticky note candidates via color segmentation."""

        regions: List[Dict[str, Any]] = []

        height, width = hsv.shape[:2]
        max_dim = max(height, width)
        scale_factor = 1.0
        if max_dim > 1400:
            scale_factor = 1400.0 / float(max_dim)
            hsv_work = cv2.resize(
                hsv,
                (int(width * scale_factor), int(height * scale_factor)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            hsv_work = hsv

        kernel, close_iter, open_iter = self._get_morphology_settings(
            hsv_work.shape, base_kernel=15, base_close=4, base_open=2
        )

        for color_name, (lower, upper) in OCR_COLOR_RANGES.items():
            lower_bound = lower.copy()
            upper_bound = upper.copy()

            # Expand hue range based on configured tolerance
            lower_bound[0] = max(0, lower_bound[0] - self.hsv_tolerance)
            upper_bound[0] = min(180, upper_bound[0] + self.hsv_tolerance)

            mask = cv2.inRange(hsv_work, lower_bound, upper_bound)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue

                area_original = area / (scale_factor**2)
                if area_original < self.min_size or area_original > self.max_size:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if scale_factor != 1.0:
                    x = int(round(x / scale_factor))
                    y = int(round(y / scale_factor))
                    w = int(round(w / scale_factor))
                    h = int(round(h / scale_factor))

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                w = max(1, min(width - x, w))
                h = max(1, min(height - y, h))

                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    continue

                regions.append(
                    {
                        "bbox": (x, y, w, h),
                        "color_name": color_name,
                        "area": area_original,
                        "source": "color",
                    }
                )

        return regions

    def _prepare_hsv_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply smoothing and contrast normalization before HSV conversion."""

        # Bilateral filter preserves edges while smoothing gradients
        blurred = cv2.bilateralFilter(image, d=9, sigmaColor=60, sigmaSpace=60)

        # Normalize luminance to reduce shading differences
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_chan)
        lab_eq = cv2.merge((l_eq, a_chan, b_chan))
        balanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Soft blur to further smooth pastel transitions
        balanced_bgr = cv2.GaussianBlur(balanced_bgr, (5, 5), 0)

        return cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2HSV)

    def _run_global_text_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run PaddleOCR once to gather global text boxes."""

        if ocr_engine is None:
            return []

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = ocr_engine.ocr(rgb_image)
        except Exception as exc:
            logger.error(f"Global text detection failed: {exc}")
            return []

        if not result:
            return []

        page_result = result[0]
        return self._extract_text_boxes_from_result(page_result)

    def _detect_adaptive_color_regions(
        self, image: np.ndarray, hsv: np.ndarray, seed_regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Derive HSV ranges from detected stickies and search for similar colors."""

        if len(seed_regions) < 1:
            return []

        hsv_samples: List[np.ndarray] = []
        for region in seed_regions:
            bbox = region.get("bbox")
            if not bbox:
                continue

            border_samples = self._collect_border_hsv_samples(hsv, bbox)
            if border_samples.size == 0:
                continue
            hsv_samples.append(border_samples)

        if not hsv_samples:
            return []

        sample_matrix = np.vstack(hsv_samples).astype(np.float32)
        if sample_matrix.shape[0] < 80:
            return []

        cluster_count = max(1, min(5, sample_matrix.shape[0] // 250))
        cluster_count = min(cluster_count, sample_matrix.shape[0])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        try:
            _, labels, centers = cv2.kmeans(
                sample_matrix,
                cluster_count,
                None,  # type: ignore[arg-type]
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS,
            )
        except cv2.error as exc:
            logger.debug(f"Adaptive clustering skipped: {exc}")
            return []

        labels = labels.flatten()
        kernel, close_iter, open_iter = self._get_morphology_settings(
            hsv.shape, base_kernel=13, base_close=3, base_open=1
        )
        adaptive_regions: List[Dict[str, Any]] = []
        existing_bboxes = [
            tuple(region["bbox"]) for region in seed_regions if "bbox" in region
        ]

        for cluster_idx in range(len(centers)):
            cluster_mask = labels == cluster_idx
            if not np.any(cluster_mask):
                continue

            cluster_samples = sample_matrix[cluster_mask]
            if cluster_samples.size == 0:
                continue

            center = centers[cluster_idx]
            std_dev = np.std(cluster_samples, axis=0)

            tolerance = np.array(
                [
                    max(self.hsv_tolerance, std_dev[0] * 1.5 + 5),
                    max(20.0, std_dev[1] * 1.2 + 10),
                    max(20.0, std_dev[2] * 1.2 + 10),
                ]
            )

            lower = np.clip(center - tolerance, [0, 0, 0], [180, 255, 255]).astype(
                np.uint8
            )
            upper = np.clip(center + tolerance, [0, 0, 0], [180, 255, 255]).astype(
                np.uint8
            )

            mask = cv2.inRange(hsv, lower, upper)
            if cv2.countNonZero(mask) < self.min_size // 2:
                continue

            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_size or area > self.max_size:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)

                if self._has_overlap(bbox, existing_bboxes, threshold=0.45):
                    continue

                adaptive_regions.append(
                    {
                        "bbox": bbox,
                        "color_name": f"adaptive_{cluster_idx}",
                        "area": area,
                        "source": "adaptive",
                    }
                )
                existing_bboxes.append(bbox)

        return adaptive_regions

    def _merge_candidate_regions(
        self,
        regions: List[Dict[str, Any]],
        iou_threshold: float = 0.55,
        distance_threshold: float = 45.0,
    ) -> List[Dict[str, Any]]:
        """Merge overlapping/neighboring candidate regions to avoid duplicates."""

        if len(regions) <= 1:
            return regions

        merged: List[Dict[str, Any]] = []
        sorted_regions = sorted(
            regions, key=lambda r: float(r.get("area", 0.0)), reverse=True
        )

        for region in sorted_regions:
            raw_bbox = region.get("bbox")
            if not raw_bbox or len(raw_bbox) < 4:
                continue

            bbox = (
                int(raw_bbox[0]),
                int(raw_bbox[1]),
                int(raw_bbox[2]),
                int(raw_bbox[3]),
            )

            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            merged_into_existing = False
            for existing in merged:
                existing_bbox = existing.get("bbox")
                if not existing_bbox or len(existing_bbox) < 4:
                    continue

                existing_tuple = (
                    int(existing_bbox[0]),
                    int(existing_bbox[1]),
                    int(existing_bbox[2]),
                    int(existing_bbox[3]),
                )

                iou = self._calculate_iou(bbox, existing_tuple)
                distance = self._center_distance(bbox, existing_tuple)

                if iou >= iou_threshold or distance <= distance_threshold:
                    merged_bbox = self._merge_bboxes(existing_tuple, bbox)
                    existing["bbox"] = merged_bbox
                    existing["area"] = existing["bbox"][2] * existing["bbox"][3]

                    # Prefer concrete color over adaptive when merging
                    if (
                        existing.get("color_name") == "adaptive"
                        and region.get("color_name") != "adaptive"
                    ):
                        existing["color_name"] = region.get("color_name", "adaptive")

                    # Preserve higher-confidence pre-text payloads
                    if not existing.get("pre_text") or float(
                        existing.get("pre_confidence", 0.0)
                    ) < float(region.get("pre_confidence", 0.0)):
                        existing["pre_text"] = region.get("pre_text")
                        existing["pre_confidence"] = region.get("pre_confidence")

                    merged_into_existing = True
                    break

            if not merged_into_existing:
                merged.append(region.copy())

        logger.info(
            f"Merged candidate regions from {len(regions)} down to {len(merged)} using IoU/center distance"
        )
        return merged

    def _get_morphology_settings(
        self,
        image_shape: Tuple[int, ...],
        base_kernel: int,
        base_close: int,
        base_open: int,
    ) -> Tuple[np.ndarray, int, int]:
        """Scale morphology kernel/iterations based on image size."""

        height = int(image_shape[0]) if len(image_shape) > 0 else 0
        width = int(image_shape[1]) if len(image_shape) > 1 else 0
        min_dim = max(
            1, min(height, width) if height and width else max(height, width, 1)
        )

        scale = max(0.6, min(2.5, min_dim / 800.0))
        kernel_size = int(round(base_kernel * scale))
        kernel_size = max(5, min(kernel_size, 55))
        if kernel_size % 2 == 0:
            kernel_size = max(5, kernel_size - 1)

        close_iter = max(1, min(6, int(round(base_close * scale))))
        open_iter = max(1, min(4, int(round(base_open * scale))))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return kernel, close_iter, open_iter

    def _collect_border_hsv_samples(
        self, hsv_image: np.ndarray, bbox: Tuple[int, int, int, int], border: int = 6
    ) -> np.ndarray:
        """Collect HSV samples from the border of a detected sticky."""

        x, y, w, h = bbox
        x_end = min(hsv_image.shape[1], x + w)
        y_end = min(hsv_image.shape[0], y + h)
        x = max(0, x)
        y = max(0, y)

        roi = hsv_image[y:y_end, x:x_end]
        if roi.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        border_y = max(1, min(border, roi.shape[0] // 2 if roi.shape[0] > 1 else 1))
        border_x = max(1, min(border, roi.shape[1] // 2 if roi.shape[1] > 1 else 1))

        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        mask[:border_y, :] = 1
        mask[-border_y:, :] = 1
        mask[:, :border_x] = 1
        mask[:, -border_x:] = 1

        samples = roi[mask == 1]
        return samples.reshape(-1, 3)

    def _has_overlap(
        self,
        bbox: Tuple[int, int, int, int],
        other_bboxes: List[Tuple[int, int, int, int]],
        threshold: float,
    ) -> bool:
        """Check if bbox overlaps any existing bbox beyond threshold."""

        for other in other_bboxes:
            if self._calculate_iou(bbox, other) >= threshold:
                return True
        return False

    def _should_use_paddle_fallback(
        self, regions: List[Dict[str, Any]], image_shape: Tuple[int, int, int]
    ) -> bool:
        """Decide if Paddle-based fallback detection should run."""

        if not regions:
            return True

        height, width = image_shape[:2]
        image_area = max(1, height * width)
        areas = [float(region.get("area", 0.0)) for region in regions]
        avg_area = sum(areas) / len(areas) if areas else 0.0
        coverage = avg_area / image_area

        # Trigger fallback when we barely detected anything or coverage is too small
        if len(regions) <= 2:
            return True

        return coverage < 0.002

    def _detect_regions_with_paddle(
        self,
        image: np.ndarray,
        text_boxes: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Use PaddleOCR detection boxes to propose sticky note regions."""

        if text_boxes is None or not text_boxes:
            text_boxes = self._run_global_text_detection(image)

        if not text_boxes:
            return []

        clusters = self._cluster_text_boxes(text_boxes)
        height, width = image.shape[:2]
        fallback_regions: List[Dict[str, Any]] = []

        for cluster in clusters:
            x, y, w, h = self._expand_bbox(cluster["bbox"], width, height)
            area = w * h

            if area < self.min_size * 0.5 or area > self.max_size * 1.5:
                continue

            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 5.0:
                continue

            roi = image[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            sorted_boxes = sorted(
                cluster["boxes"], key=lambda b: (b["bbox"][1], b["bbox"][0])
            )
            texts = [box["text"].strip() for box in sorted_boxes if box["text"]]

            if not texts:
                continue

            scores = [float(box.get("score", 0.0)) for box in sorted_boxes]
            avg_conf = (sum(scores) / len(scores)) if scores else 0.0
            if avg_conf <= 1.0:
                avg_conf *= 100

            fallback_regions.append(
                {
                    "bbox": (x, y, w, h),
                    "color_name": "adaptive",
                    "area": area,
                    "source": "paddle",
                    "pre_text": " ".join(texts),
                    "pre_confidence": avg_conf,
                }
            )

        return fallback_regions

    def _validate_regions_with_text(
        self,
        regions: List[Dict[str, Any]],
        text_boxes: List[Dict[str, Any]],
        image_shape: Tuple[int, int, int],
        min_coverage: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Ensure each detected region aligns with text boxes."""

        if not text_boxes:
            return regions

        validated: List[Dict[str, Any]] = []
        height, width = image_shape[:2]

        for region in regions:
            bbox = region.get("bbox")
            if not bbox:
                continue

            matching_boxes = self._filter_boxes_for_region(bbox, text_boxes)
            if not matching_boxes:
                logger.debug(f"Dropping region {bbox} (no overlapping text)")
                continue

            clusters = self._cluster_text_boxes(
                matching_boxes, iou_threshold=0.2, distance_threshold=55.0
            )

            cluster_bboxes = []
            for cluster in clusters:
                cluster_bbox = cluster.get("bbox")
                if not cluster_bbox or len(cluster_bbox) < 4:
                    continue
                cluster_bboxes.append(
                    (
                        int(cluster_bbox[0]),
                        int(cluster_bbox[1]),
                        int(cluster_bbox[2]),
                        int(cluster_bbox[3]),
                    )
                )

            if not cluster_bboxes:
                logger.debug(f"Region {bbox} clusters missing bbox data")
                continue

            combined_bbox: Tuple[int, int, int, int] = cluster_bboxes[0]
            for cluster_bbox in cluster_bboxes[1:]:
                combined_bbox = self._merge_bboxes(combined_bbox, cluster_bbox)

            region_area = bbox[2] * bbox[3]
            combined_area = combined_bbox[2] * combined_bbox[3]
            coverage = combined_area / region_area if region_area > 0 else 0.0

            pad_ratio = 0.12 if coverage >= min_coverage else 0.08
            new_bbox = self._expand_bbox(
                combined_bbox, width, height, pad_ratio=pad_ratio
            )

            new_area = new_bbox[2] * new_bbox[3]
            if new_area < self.min_size * 0.4:
                logger.debug(f"Region {bbox} rejected after merge due to tiny area")
                continue

            new_region = region.copy()
            new_region["bbox"] = new_bbox
            new_region["area"] = new_area
            validated.append(new_region)

        logger.info(
            f"Text alignment validation kept {len(validated)} regions out of {len(regions)}"
        )
        return validated

    def _filter_boxes_for_region(
        self, bbox: Tuple[int, int, int, int], text_boxes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return text boxes overlapping with the provided region."""

        rx, ry, rw, rh = bbox
        filtered: List[Dict[str, Any]] = []

        for box in text_boxes:
            box_bbox = box.get("bbox")
            if not box_bbox:
                continue

            if self._calculate_iou(bbox, box_bbox) >= 0.08:
                filtered.append(box)
                continue

            center_x = box_bbox[0] + box_bbox[2] / 2.0
            center_y = box_bbox[1] + box_bbox[3] / 2.0

            if rx <= center_x <= rx + rw and ry <= center_y <= ry + rh:
                filtered.append(box)

        return filtered

    def _compose_text_from_boxes(
        self, bbox: Tuple[int, int, int, int], text_boxes: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Combine text from global OCR boxes that fall inside bbox."""

        matching_boxes = self._filter_boxes_for_region(bbox, text_boxes)
        if not matching_boxes:
            return "", 0.0

        ordered = sorted(matching_boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        texts = []
        for box in ordered:
            text_value = box.get("text")
            if not text_value:
                continue
            texts.append(
                text_value.strip() if isinstance(text_value, str) else str(text_value)
            )
        if not texts:
            return "", 0.0

        scores = [self._safe_float(box.get("score")) for box in ordered]
        avg_conf = sum(scores) / len(scores) if scores else 0.0
        if avg_conf <= 1.0:
            avg_conf *= 100

        return " ".join(texts).strip(), avg_conf

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convert value to float, guarding against None or invalid inputs."""

        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_text_boxes_from_result(self, page_result: Any) -> List[Dict[str, Any]]:
        """Normalize PaddleOCR detection output into text boxes."""

        boxes: List[Dict[str, Any]] = []

        if isinstance(page_result, dict):
            texts = page_result.get("rec_texts", []) or []
            scores = page_result.get("rec_scores", []) or []
            polys = (
                page_result.get("rec_polys")
                or page_result.get("det_polys")
                or page_result.get("polys")
                or page_result.get("det_boxes")
                or []
            )

            for idx, poly in enumerate(polys):
                bbox = self._poly_to_bbox(poly)
                if not bbox:
                    continue

                text = texts[idx] if idx < len(texts) else ""
                score = scores[idx] if idx < len(scores) else 0.0

                boxes.append({"bbox": bbox, "text": text, "score": float(score)})
        else:
            for entry in page_result or []:
                if not entry or len(entry) < 2:
                    continue

                poly = entry[0]
                meta = entry[1]
                bbox = self._poly_to_bbox(poly)
                if not bbox or not isinstance(meta, (list, tuple)) or len(meta) < 2:
                    continue

                text = str(meta[0])
                score = float(meta[1]) if meta[1] is not None else 0.0
                boxes.append({"bbox": bbox, "text": text, "score": score})

        return boxes

    def _poly_to_bbox(self, poly: Any) -> Optional[Tuple[int, int, int, int]]:
        """Convert PaddleOCR polygon formats to bounding box."""

        if poly is None:
            return None

        points: List[Tuple[float, float]] = []

        if isinstance(poly, (list, tuple, np.ndarray)):
            iterable = list(poly)

            if len(iterable) == 4 and all(
                isinstance(v, (int, float)) for v in iterable
            ):
                x1, y1, x2, y2 = iterable
                return (
                    int(min(x1, x2)),
                    int(min(y1, y2)),
                    int(abs(x2 - x1)),
                    int(abs(y2 - y1)),
                )

            for item in iterable:
                if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                    points.append((float(item[0]), float(item[1])))

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = int(max(1, x_max - x_min))
        height = int(max(1, y_max - y_min))
        return (int(x_min), int(y_min), width, height)

    def _cluster_text_boxes(
        self,
        text_boxes: List[Dict[str, Any]],
        iou_threshold: float = 0.35,
        distance_threshold: float = 80.0,
    ) -> List[Dict[str, Any]]:
        """Cluster nearby text boxes into sticky-sized groups."""

        clusters: List[Dict[str, Any]] = []

        for box in text_boxes:
            assigned = False

            for cluster in clusters:
                iou = self._calculate_iou(box["bbox"], cluster["bbox"])
                distance = self._center_distance(box["bbox"], cluster["bbox"])

                if iou >= iou_threshold or distance <= distance_threshold:
                    cluster["boxes"].append(box)
                    cluster["bbox"] = self._merge_bboxes(cluster["bbox"], box["bbox"])
                    assigned = True
                    break

            if not assigned:
                clusters.append({"boxes": [box], "bbox": box["bbox"]})

        return clusters

    def _merge_bboxes(
        self, bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Create a bounding box that covers both inputs."""

        x1 = min(bbox_a[0], bbox_b[0])
        y1 = min(bbox_a[1], bbox_b[1])
        x2 = max(bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2])
        y2 = max(bbox_a[1] + bbox_a[3], bbox_b[1] + bbox_b[3])
        return (x1, y1, x2 - x1, y2 - y1)

    def _center_distance(
        self, bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Euclidean distance between bbox centers."""

        center_a = (bbox_a[0] + bbox_a[2] / 2.0, bbox_a[1] + bbox_a[3] / 2.0)
        center_b = (bbox_b[0] + bbox_b[2] / 2.0, bbox_b[1] + bbox_b[3] / 2.0)
        return float(
            np.sqrt((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2)
        )

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        max_width: int,
        max_height: int,
        pad_ratio: float = 0.08,
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box slightly to include sticky background."""

        x, y, w, h = bbox
        pad_w = int(w * pad_ratio)
        pad_h = int(h * pad_ratio)

        x_new = max(0, x - pad_w)
        y_new = max(0, y - pad_h)
        w_new = min(max_width - x_new, w + pad_w * 2)
        h_new = min(max_height - y_new, h + pad_h * 2)

        return (x_new, y_new, max(1, w_new), max(1, h_new))

    def _get_dominant_color(self, roi: np.ndarray) -> str:
        """
        Extract dominant color from ROI using k-means clustering.

        Args:
            roi: Region of interest (BGR image)

        Returns:
            Hex color string (e.g., '#FFFF00')
        """
        # Reshape to 2D array of pixels
        pixels = roi.reshape(-1, 3)
        pixels = np.float32(pixels)

        # K-means clustering (k=1 for dominant color)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(
            pixels,  # type: ignore[arg-type]
            1,
            None,  # type: ignore[arg-type]
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        # Convert BGR to RGB
        dominant_color = centers[0].astype(int)
        r, g, b = int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])

        return f"#{r:02x}{g:02x}{b:02x}"

    def _extract_text(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from ROI using PaddleOCR with preprocessing.

        Args:
            roi: Region of interest (BGR image)

        Returns:
            Tuple of (text, confidence)
        """
        if ocr_engine is None:
            logger.error("PaddleOCR not initialized")
            return "", 0.0

        try:
            # Check if background is dark (black sticky note with white text)
            # Calculate mean brightness of the ROI
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray_roi))  # type: ignore[arg-type]

            # If background is dark (mean < 100), invert for better OCR
            if mean_brightness < 100:
                logger.info(
                    f"Dark background detected (brightness={mean_brightness:.1f}), inverting image for OCR"
                )
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                rgb_roi = 255 - rgb_roi  # Invert colors: black→white, white→black
            else:
                # PaddleOCR expects RGB format
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Run OCR (removed cls parameter - no longer supported in newer PaddleOCR)
            result = ocr_engine.ocr(rgb_roi)

            # Check if result is valid
            if not result:
                logger.warning("OCR returned empty result")
                return "", 0.0

            # PaddleOCR 3.x returns a list with one dict containing results
            if not isinstance(result, list) or len(result) == 0:
                logger.warning("OCR result is not a list or is empty")
                return "", 0.0

            page_result = result[0]
            if not page_result or page_result == 0 or page_result is None:
                logger.warning("OCR page result is empty")
                return "", 0.0

            # PaddleOCR 3.x format: dictionary with keys rec_texts, rec_scores, rec_polys
            if isinstance(page_result, dict):
                rec_texts = page_result.get("rec_texts", [])
                rec_scores = page_result.get("rec_scores", [])

                if not rec_texts:
                    logger.warning("No text detected (rec_texts is empty)")
                    return "", 0.0

                # Combine texts and calculate average confidence
                combined_text = " ".join(rec_texts)
                avg_confidence = (
                    sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
                )

                # Convert to percentage (0-100) if needed
                if avg_confidence <= 1.0:
                    avg_confidence = avg_confidence * 100

                logger.info(
                    f"Extracted {len(rec_texts)} text segments, avg confidence: {avg_confidence:.1f}%"
                )
                return combined_text.strip(), avg_confidence
            else:
                # Fallback: older PaddleOCR format [[[box], (text, confidence)], ...]
                logger.warning("Using fallback parsing for older PaddleOCR format")
                texts = []
                confidences = []

                for line in page_result:
                    if not line:
                        continue

                    try:
                        if (
                            len(line) >= 2
                            and isinstance(line[1], (tuple, list))
                            and len(line[1]) >= 2
                        ):
                            text = str(line[1][0])
                            conf = float(line[1][1])
                            texts.append(text)
                            confidences.append(conf)
                    except (TypeError, ValueError, IndexError):
                        continue

                combined_text = " ".join(texts)
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )

                if avg_confidence <= 1.0:
                    avg_confidence = avg_confidence * 100

                return combined_text.strip(), avg_confidence

        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return "", 0.0

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for better OCR accuracy.

        Pipeline:
        1. Grayscale conversion
        2. Gaussian blur (noise reduction)
        3. Sharpening
        4. Adaptive thresholding

        Args:
            roi: Region of interest (BGR image)

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.

        Args:
            bbox1: First bounding box (x, y, width, height)
            bbox2: Second bounding box (x, y, width, height)

        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        # Check if there's no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou

    def _consolidate_colors(
        self, regions: List[StickyRegion], color_distance_threshold: int = 30
    ) -> List[StickyRegion]:
        """
        Consolidate similar colors to prevent duplicate color mappings.
        Groups regions with similar hex colors and assigns them the same representative color.

        Args:
            regions: List of StickyRegion objects
            color_distance_threshold: Maximum RGB distance to consider colors as similar (0-255 scale)

        Returns:
            List of regions with consolidated colors
        """
        if len(regions) <= 1:
            return regions

        def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            """Convert hex color to RGB tuple."""
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore

        def rgb_distance(
            rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]
        ) -> float:
            """Calculate Euclidean distance between two RGB colors."""
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

        # Group regions by similar colors
        color_groups: List[List[StickyRegion]] = []
        processed = set()

        for i, region in enumerate(regions):
            if i in processed:
                continue

            # Start new color group
            group = [region]
            processed.add(i)
            rgb1 = hex_to_rgb(region.color_hex)

            # Find all similar colors
            for j, other_region in enumerate(regions):
                if j in processed:
                    continue

                rgb2 = hex_to_rgb(other_region.color_hex)
                distance = rgb_distance(rgb1, rgb2)

                if distance <= color_distance_threshold:
                    group.append(other_region)
                    processed.add(j)

            color_groups.append(group)

        # Assign representative color to each group (use most common color)
        consolidated_regions = []
        for group in color_groups:
            # Use the color from the first region as representative
            representative_color = group[0].color_hex
            representative_name = group[0].color_name

            # Update all regions in group to use representative color
            for region in group:
                region.color_hex = representative_color
                region.color_name = representative_name
                consolidated_regions.append(region)

        logger.info(
            f"Consolidated {len(regions)} regions into {len(color_groups)} color groups"
        )
        return consolidated_regions

    def _remove_duplicates(
        self, regions: List[StickyRegion], iou_threshold: float = 0.5
    ) -> List[StickyRegion]:
        """
        Remove duplicate regions based on bounding box overlap.
        When regions overlap significantly, keep the one with more text content.

        Args:
            regions: List of StickyRegion objects
            iou_threshold: IoU threshold for duplicates (default: 0.5 = 50% overlap)

        Returns:
            List of deduplicated StickyRegion objects
        """
        if len(regions) <= 1:
            return regions

        # Sort regions by text length first (prefer regions with more text), then confidence
        sorted_regions = sorted(
            regions, key=lambda r: (len(r.text), r.confidence), reverse=True
        )

        # Keep track of regions to remove
        to_remove = set()

        # Compare each region with others
        for i, region_a in enumerate(sorted_regions):
            if i in to_remove:
                continue

            for j, region_b in enumerate(sorted_regions):
                if i == j or j in to_remove:
                    continue

                # Calculate IoU
                iou = self._calculate_iou(region_a.bbox, region_b.bbox)

                # If ANY overlap (even 15%), merge them - prioritize complete text extraction
                if iou > iou_threshold:
                    # Merge text if region_b has additional text not in region_a
                    if region_b.text and region_b.text not in region_a.text:
                        # Combine texts
                        combined = f"{region_a.text} {region_b.text}".strip()
                        region_a.text = combined
                        logger.debug(
                            f"Merging text from region {region_b.id} into {region_a.id}"
                        )

                    to_remove.add(j)
                    logger.debug(
                        f"Removing duplicate region {region_b.id} (IoU={iou:.2f} with region {region_a.id})"
                    )

        # Filter out removed regions
        unique_regions = [r for i, r in enumerate(sorted_regions) if i not in to_remove]

        # Re-assign sequential IDs
        for new_id, region in enumerate(unique_regions):
            region.id = new_id

        logger.info(f"Removed {len(to_remove)} duplicate regions out of {len(regions)}")
        return unique_regions

    def _link_regions(self, regions: List[StickyRegion]) -> None:
        """
        Link regions based on spatial proximity.
        Finds description stickies near summary stickies.

        Args:
            regions: List of StickyRegion objects (modified in-place)
        """
        for i, region_a in enumerate(regions):
            for j, region_b in enumerate(regions):
                if i == j:
                    continue

                # Calculate distance between centers
                center_a = (
                    region_a.bbox[0] + region_a.bbox[2] // 2,
                    region_a.bbox[1] + region_a.bbox[3] // 2,
                )
                center_b = (
                    region_b.bbox[0] + region_b.bbox[2] // 2,
                    region_b.bbox[1] + region_b.bbox[3] // 2,
                )

                distance = np.sqrt(
                    (center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2
                )

                # Link if within proximity threshold
                if distance < self.proximity:
                    if region_b.id not in region_a.linked_to:
                        region_a.linked_to.append(region_b.id)

    def _region_to_dict(self, region: StickyRegion) -> Dict[str, Any]:
        """Convert StickyRegion to dictionary format."""
        return {
            "id": region.id,
            "color_hex": region.color_hex,
            "color_name": region.color_name,
            "text": region.text,
            "bbox": {
                "x": region.bbox[0],
                "y": region.bbox[1],
                "width": region.bbox[2],
                "height": region.bbox[3],
            },
            "confidence": round(region.confidence, 2),
            "linked_to": region.linked_to,
        }


# ============================================================================
# Background Processing Helper
# ============================================================================


def process_image_async(
    image_path: str, socketio, callback_event: str = "ocr_progress", **kwargs
) -> List[Dict[str, Any]]:
    """
    Process image in background thread with SocketIO progress reporting.

    Args:
        image_path: Path to input image
        socketio: Flask-SocketIO instance
        callback_event: SocketIO event name for progress
        **kwargs: Additional OCR parameters

    Returns:
        List of region dictionaries
    """

    def progress_callback(current, total, status, message, preview_url=None):
        """Emit progress via SocketIO."""
        payload = {
            "percent": int((current / total) * 100) if total > 0 else 0,
            "current_region": current,
            "total_regions": total,
            "status": status,
            "message": message,
        }
        if preview_url:
            payload["preview_url"] = preview_url

        socketio.emit(callback_event, payload)

    # Create OCR service
    ocr_service = OCRService(
        hsv_tolerance=kwargs.get("hsv_tolerance", DEFAULT_HSV_TOLERANCE),
        min_size=kwargs.get("min_size", DEFAULT_MIN_SIZE),
        max_size=kwargs.get("max_size", DEFAULT_MAX_SIZE),
        proximity=kwargs.get("proximity", DEFAULT_PROXIMITY),
    )

    # Process image
    try:
        regions = ocr_service.process_image(image_path, progress_callback)

        # Save regions to database for persistence
        from services import session_manager
        import os

        image_filename = os.path.basename(image_path)
        saved_regions = []

        for region in regions:
            # Prepare issue data for database
            issue_data = {
                "image_filename": image_filename,
                "region_id": region["id"],
                "color_hex": region["color_hex"],
                "summary": region["text"],
                "description": "",  # Will be filled from linked regions
                "issue_type": "",  # Will be set during mapping
                "project_key": "",  # Will be set during mapping
                "issue_key": None,  # Will be set after Jira import
                "confidence": region["confidence"],
                "bbox_x": region["bbox"]["x"],
                "bbox_y": region["bbox"]["y"],
                "bbox_width": region["bbox"]["width"],
                "bbox_height": region["bbox"]["height"],
            }

            # Save to database and get ID
            db_id = session_manager.create_issue(issue_data)
            region["db_id"] = db_id  # Add database ID to region
            saved_regions.append(region)

        logger.info(f"Saved {len(saved_regions)} regions to database")

        # Emit completion with database IDs included
        socketio.emit(
            callback_event,
            {
                "percent": 100,
                "status": "complete",
                "message": f"Extracted {len(saved_regions)} regions",
                "regions": saved_regions,
            },
        )

        return saved_regions
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}", exc_info=True)

        # Emit error
        socketio.emit(callback_event, {"status": "error", "message": str(e)})

        raise


# ============================================================================
# Convenience Factory Function
# ============================================================================


def create_ocr_service(**kwargs) -> OCRService:
    """
    Factory function to create OCRService instance.

    Args:
        **kwargs: OCR parameters (hsv_tolerance, min_size, max_size, proximity)

    Returns:
        OCRService instance
    """
    return OCRService(**kwargs)
