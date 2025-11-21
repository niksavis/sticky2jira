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

# HSV color ranges for common sticky note colors
# Format: (lower_bound, upper_bound) in HSV color space
OCR_COLOR_RANGES = {
    "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    "pink": (np.array([140, 50, 50]), np.array([170, 255, 255])),
    "blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),
    "green": (np.array([40, 50, 50]), np.array([80, 255, 255])),
    "orange": (np.array([10, 100, 100]), np.array([20, 255, 255])),
}

# Default processing parameters
DEFAULT_HSV_TOLERANCE = 20
DEFAULT_MIN_SIZE = 50
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

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect regions for each color
        regions = []
        region_id = 0

        for color_name, (lower, upper) in OCR_COLOR_RANGES.items():
            # Adjust bounds with tolerance
            lower_adjusted = np.maximum(lower - self.hsv_tolerance, 0)
            upper_adjusted = np.minimum(upper + self.hsv_tolerance, 255)

            # Create color mask
            mask = cv2.inRange(hsv, lower_adjusted, upper_adjusted)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by size
                if area < self.min_size or area > self.max_size:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Extract ROI
                roi = image[y : y + h, x : x + w]

                # Get dominant color
                color_hex = self._get_dominant_color(roi)

                # Preprocess and extract text
                text, confidence = self._extract_text(roi)

                # Skip empty text
                if not text.strip():
                    continue

                # Create region
                region = StickyRegion(
                    id=region_id,
                    color_hex=color_hex,
                    color_name=color_name,
                    text=text,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    linked_to=[],
                )
                regions.append(region)
                region_id += 1

                # Report progress
                if progress_callback:
                    progress_callback(
                        current=region_id,
                        total=region_id,  # Updated dynamically
                        status="processing",
                        message=f"Extracted text from {color_name} sticky note",
                    )

        # Spatial linking (find description stickies near summary stickies)
        self._link_regions(regions)

        # Convert to dictionary format
        result = [self._region_to_dict(r) for r in regions]

        logger.info(f"OCR processing complete: {len(result)} regions extracted")
        return result

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

        # Emit completion
        socketio.emit(
            callback_event,
            {
                "percent": 100,
                "status": "complete",
                "message": f"Extracted {len(regions)} regions",
                "regions": regions,
            },
        )

        return regions
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
