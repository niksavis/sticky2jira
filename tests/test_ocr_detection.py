"""
Pytest tests for OCR sticky note detection with visual debugging outputs.

This test suite validates the OCR detection pipeline and generates debug visualizations
to help diagnose detection issues without running the full Flask app.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from services.ocr_service import OCRService


# Test fixtures and configuration
@pytest.fixture
def test_image_path():
    """Path to the test sticky notes image."""
    return "test_images/sticky_notes_sample.png"


@pytest.fixture
def debug_output_dir():
    """Directory for debug output images."""
    output_dir = Path("test_output/debug_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def ocr_service():
    """Create OCR service instance with default parameters."""
    return OCRService(hsv_tolerance=20, min_size=1500, max_size=50000, proximity=100)


class TestOCRDetection:
    """Test suite for OCR sticky note detection."""

    # Expected text from sticky notes (ground truth)
    EXPECTED_TEXTS = [
        "This is a sticky in grey",
        "This is a sticky in yellow 1",
        "This is a sticky in yellow 2",
        "This is a sticky in orange",
        "This is a sticky in green 1",
        "This is a sticky in green 2",
        "This is a sticky in green 3",
        "This is a sticky in pink 1",
        "This is a sticky in red",
        "This is a sticky in blue 3",
        "This is a sticky in blue 2",
        "This is a sticky in blue 1",
        "This is a sticky in black",
        "This is a sticky in violet",
        "This is a sticky in teal blue",
        "This is a sticky in pink 2",
    ]

    def test_image_exists(self, test_image_path):
        """Verify test image exists and is readable."""
        image = cv2.imread(test_image_path)
        assert image is not None, f"Could not load test image: {test_image_path}"
        assert image.shape[0] > 0 and image.shape[1] > 0, "Invalid image dimensions"

    def test_ocr_detection_count(self, ocr_service, test_image_path):
        """Test that OCR detects exactly 16 regions (one per sticky note)."""
        regions = ocr_service.process_image(test_image_path)

        assert len(regions) == 16, (
            f"Expected exactly 16 sticky notes, detected {len(regions)}. "
            f"This suggests clustering parameters may need adjustment."
        )

    def test_regions_have_text(self, ocr_service, test_image_path):
        """Verify all detected regions contain text."""
        regions = ocr_service.process_image(test_image_path)

        for i, region in enumerate(regions):
            assert region["text"], f"Region {i} has no text"
            assert len(region["text"].strip()) > 0, f"Region {i} has empty text"

    def test_regions_have_valid_bboxes(self, ocr_service, test_image_path):
        """Verify all regions have valid bounding boxes."""
        regions = ocr_service.process_image(test_image_path)
        image = cv2.imread(test_image_path)
        height, width = image.shape[:2]

        for i, region in enumerate(regions):
            bbox = region["bbox"]
            assert "x" in bbox and "y" in bbox, f"Region {i} missing bbox coordinates"
            assert "width" in bbox and "height" in bbox, (
                f"Region {i} missing bbox dimensions"
            )

            # Verify bbox is within image bounds
            assert 0 <= bbox["x"] < width, f"Region {i} x out of bounds"
            assert 0 <= bbox["y"] < height, f"Region {i} y out of bounds"
            assert bbox["width"] > 0, f"Region {i} has zero/negative width"
            assert bbox["height"] > 0, f"Region {i} has zero/negative height"

    def test_regions_have_confidence(self, ocr_service, test_image_path):
        """Verify confidence scores are in valid range."""
        regions = ocr_service.process_image(test_image_path)

        for i, region in enumerate(regions):
            confidence = region["confidence"]
            assert 0 <= confidence <= 100, (
                f"Region {i} confidence {confidence} out of range [0, 100]"
            )

    def test_exact_text_extraction(self, ocr_service, test_image_path):
        """Test that all expected texts are extracted with high accuracy.

        Note: PaddleOCR may have minor character recognition errors (<5% of text),
        especially at image edges. This test allows for close matches (>80% similarity).
        """
        regions = ocr_service.process_image(test_image_path)
        detected_texts = [region["text"] for region in regions]

        # Check that all expected texts are found (exact match or very close)
        missing_texts = []
        fuzzy_matches = {}

        for expected in self.EXPECTED_TEXTS:
            if expected not in detected_texts:
                # Check for fuzzy match (OCR errors like "ticky" instead of "sticky")
                fuzzy_match = None
                for detected in detected_texts:
                    # Calculate similarity - allow if >80% of characters match
                    if self._text_similarity(expected, detected) > 0.8:
                        fuzzy_match = detected
                        break

                if fuzzy_match:
                    fuzzy_matches[expected] = fuzzy_match
                else:
                    missing_texts.append(expected)

        if missing_texts or fuzzy_matches:
            print("\n=== TEXT EXTRACTION VALIDATION ===")
            print(
                f"Expected {len(self.EXPECTED_TEXTS)} texts, found {len(detected_texts)}"
            )

            if fuzzy_matches:
                print("\nFuzzy matches (minor OCR errors):")
                for expected, detected in fuzzy_matches.items():
                    print(f"  Expected: '{expected}'")
                    print(
                        f"  Detected: '{detected}' (similarity: {self._text_similarity(expected, detected):.1%})"
                    )

            if missing_texts:
                print(f"\nCompletely missing {len(missing_texts)} texts:")
                for text in missing_texts:
                    print(f"  - {text}")
                assert False, (
                    f"Missing {len(missing_texts)} expected texts with no fuzzy matches"
                )

    def _text_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio using Levenshtein distance."""
        if not str1 or not str2:
            return 0.0

        # Calculate Levenshtein distance
        len1, len2 = len(str1), len(str2)
        if len1 < len2:
            str1, str2 = str2, str1
            len1, len2 = len2, len1

        # Create distance matrix
        previous_row = range(len2 + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        distance = previous_row[-1]
        max_len = max(len1, len2)

        # Return similarity as 1 - (distance / max_length)
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def test_no_extra_regions(self, ocr_service, test_image_path):
        """Test that no extra/spurious regions are detected.

        Note: Allows fuzzy matches for OCR variations (e.g., "ticky" vs "sticky").
        """
        regions = ocr_service.process_image(test_image_path)
        detected_texts = [region["text"] for region in regions]

        # All detected texts should match expected (exact or fuzzy)
        unexpected = []
        for detected in detected_texts:
            if detected not in self.EXPECTED_TEXTS:
                # Check for fuzzy match
                fuzzy_matched = False
                for expected in self.EXPECTED_TEXTS:
                    if self._text_similarity(expected, detected) > 0.8:
                        fuzzy_matched = True
                        break

                if not fuzzy_matched:
                    unexpected.append(detected)

        if unexpected:
            print("\n=== SPURIOUS REGIONS DETECTED ===")
            print(f"Found {len(unexpected)} completely unexpected regions:")
            for text in unexpected:
                print(f"  - {text}")
            assert False, (
                f"Found {len(unexpected)} spurious regions with no fuzzy matches"
            )


class TestOCRDebugging:
    """Test suite that generates visual debugging outputs."""

    def test_visualize_detected_regions(
        self, ocr_service, test_image_path, debug_output_dir
    ):
        """Generate visualization showing all detected regions with bounding boxes."""
        regions = ocr_service.process_image(test_image_path)
        image = cv2.imread(test_image_path)

        # Create visualization
        vis_image = image.copy()
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
            (192, 0, 0),
            (0, 192, 0),
            (0, 0, 192),
            (192, 192, 0),
            (192, 0, 192),
            (0, 192, 192),
            (64, 0, 0),
            (0, 64, 0),
        ]

        for i, region in enumerate(regions):
            bbox = region["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            color = colors[i % len(colors)]

            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw region number
            cv2.putText(
                vis_image,
                str(i + 1),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Draw text preview (first 15 chars)
            text_preview = (
                region["text"][:15] + "..."
                if len(region["text"]) > 15
                else region["text"]
            )
            cv2.putText(
                vis_image,
                text_preview,
                (x + 5, y + h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Save visualization
        output_path = debug_output_dir / "detected_regions.png"
        cv2.imwrite(str(output_path), vis_image)

        print(f"\n{'=' * 70}")
        print(f"Detected {len(regions)} regions")
        print(f"Visualization saved to: {output_path}")
        print(f"{'=' * 70}")

        for i, region in enumerate(regions):
            print(f"{i + 1}. [{region['color_name']}] {region['text'][:60]}")

        assert output_path.exists(), "Visualization file was not created"

    def test_visualize_color_masks(self, test_image_path, debug_output_dir):
        """Generate color segmentation masks for each sticky color."""
        from services.ocr_service import OCR_COLOR_RANGES

        image = cv2.imread(test_image_path)

        # Apply preprocessing
        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed = cv2.GaussianBlur(filtered, (5, 5), 0)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        # Create combined visualization
        height, width = image.shape[:2]
        vis_height = height * 3
        vis_width = width * 3
        combined_vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

        color_idx = 0
        for color_name, (lower, upper) in OCR_COLOR_RANGES.items():
            if color_name in ["gray", "black"]:
                continue

            # Create mask
            lower_bound = lower.copy()
            upper_bound = upper.copy()
            lower_bound[0] = max(0, lower_bound[0] - 15)
            upper_bound[0] = min(180, upper_bound[0] + 15)
            lower_bound[1] = max(0, lower_bound[1] - 10)

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

            # Convert mask to BGR for visualization
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Add to combined visualization (3x3 grid)
            row = color_idx // 3
            col = color_idx % 3

            if row < 3 and col < 3:
                y_start = row * height
                x_start = col * width
                combined_vis[y_start : y_start + height, x_start : x_start + width] = (
                    mask_bgr
                )

                # Add label
                cv2.putText(
                    combined_vis,
                    color_name,
                    (x_start + 10, y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Save individual mask
            mask_path = debug_output_dir / f"mask_{color_name}.png"
            cv2.imwrite(str(mask_path), mask)

            color_idx += 1

        # Save combined visualization
        combined_path = debug_output_dir / "color_masks_combined.png"
        cv2.imwrite(str(combined_path), combined_vis)

        print(f"\nColor mask visualizations saved to: {debug_output_dir}")
        print(f"Combined view: {combined_path}")

        assert combined_path.exists(), "Combined mask visualization not created"

    def test_visualize_text_clustering(self, test_image_path, debug_output_dir):
        """Visualize text box clustering at different distance thresholds."""
        from paddleocr import PaddleOCR

        image = cv2.imread(test_image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run OCR
        ocr = PaddleOCR(use_textline_orientation=True, lang="en")
        result = ocr.ocr(rgb_image)

        if not result or not result[0]:
            pytest.skip("No OCR results returned")

        ocr_result = result[0]

        # Extract text boxes
        text_boxes = []
        rec_texts = ocr_result.get("rec_texts", [])
        rec_scores = ocr_result.get("rec_scores", [])
        rec_polys = ocr_result.get("rec_polys", [])

        for i in range(len(rec_texts)):
            poly = rec_polys[i]
            points = np.array(poly)
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            text_boxes.append(
                {"bbox": bbox, "text": rec_texts[i], "score": rec_scores[i]}
            )

        print(f"\nTotal text boxes detected: {len(text_boxes)}")

        # Test different clustering thresholds
        thresholds = [20, 30, 45, 60]

        for threshold in thresholds:
            vis_image = image.copy()

            # Simple clustering
            clusters = self._cluster_text_boxes(text_boxes, threshold)

            # Visualize clusters
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 0, 0),
                (0, 128, 0),
                (0, 0, 128),
                (128, 128, 0),
                (128, 0, 128),
                (0, 128, 128),
                (192, 0, 0),
                (0, 192, 0),
                (0, 0, 192),
                (192, 192, 0),
            ]

            for i, cluster in enumerate(clusters):
                color = colors[i % len(colors)]

                for box in cluster["boxes"]:
                    x, y, w, h = box["bbox"]
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)

                # Draw cluster number at first box
                if cluster["boxes"]:
                    first_box = cluster["boxes"][0]["bbox"]
                    cv2.putText(
                        vis_image,
                        str(i + 1),
                        (first_box[0], first_box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # Save visualization
            vis_path = debug_output_dir / f"clustering_threshold_{threshold}px.png"
            cv2.imwrite(str(vis_path), vis_image)

            print(
                f"Threshold {threshold}px: {len(clusters)} clusters -> {vis_path.name}"
            )

        assert len(list(debug_output_dir.glob("clustering_threshold_*.png"))) == len(
            thresholds
        )

    def _cluster_text_boxes(self, text_boxes, distance_threshold=45.0):
        """Simple clustering for visualization."""
        clusters = []

        for box in text_boxes:
            assigned = False
            box_center = (
                box["bbox"][0] + box["bbox"][2] / 2,
                box["bbox"][1] + box["bbox"][3] / 2,
            )

            for cluster in clusters:
                for c_box in cluster["boxes"]:
                    c_center = (
                        c_box["bbox"][0] + c_box["bbox"][2] / 2,
                        c_box["bbox"][1] + c_box["bbox"][3] / 2,
                    )
                    dist = np.sqrt(
                        (box_center[0] - c_center[0]) ** 2
                        + (box_center[1] - c_center[1]) ** 2
                    )

                    if dist < distance_threshold:
                        cluster["boxes"].append(box)
                        assigned = True
                        break

                if assigned:
                    break

            if not assigned:
                clusters.append({"boxes": [box]})

        return clusters

    def test_generate_comparison_report(
        self, ocr_service, test_image_path, debug_output_dir
    ):
        """Generate a comprehensive comparison report."""
        regions = ocr_service.process_image(test_image_path)

        report_path = debug_output_dir / "detection_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("OCR DETECTION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test Image: {test_image_path}\n")
            f.write(f"Total Regions Detected: {len(regions)}\n")
            f.write("Expected Regions: 16 sticky notes\n")
            f.write(
                f"Detection Accuracy: {'PASS' if len(regions) == 16 else 'NEEDS TUNING'}\n\n"
            )

            f.write("=" * 70 + "\n")
            f.write("DETECTED REGIONS\n")
            f.write("=" * 70 + "\n\n")

            for i, region in enumerate(regions):
                f.write(f"Region {i + 1}:\n")
                f.write(f"  Color: {region['color_name']} ({region['color_hex']})\n")
                f.write(f"  Text: {region['text']}\n")
                f.write(f"  Confidence: {region['confidence']:.1f}%\n")
                f.write(
                    f"  BBox: ({region['bbox']['x']}, {region['bbox']['y']}) "
                    f"{region['bbox']['width']}x{region['bbox']['height']}\n"
                )
                f.write("\n")

        print(f"\nDetection report saved to: {report_path}")

        assert report_path.exists(), "Report file was not created"


class TestRealWorldScenarios:
    """Test suite for real-world sticky note scenarios."""

    def test_story_bug_task_detection(self, ocr_service):
        """Test typical 3-color workflow (Story=green, Bug=red, Task=cyan)."""
        image_path = "test_images/story_bug_task_sample.png"

        # Verify image exists
        image = cv2.imread(image_path)
        assert image is not None, f"Could not load image: {image_path}"

        regions = ocr_service.process_image(image_path)

        # Expected: 14 stickies total (4 stories, 5 tasks, 5 bugs based on analysis)
        assert len(regions) == 14, (
            f"Expected 14 stickies in story/bug/task image, detected {len(regions)}"
        )

        # Verify all regions have text
        for region in regions:
            assert region["text"], "Region missing text"
            assert region["confidence"] > 85, (
                f"Low confidence: {region['confidence']:.1f}% for '{region['text']}'"
            )

        # Verify color distribution (should be 3 colors: red, lime/green, cyan)
        colors = [r["color_name"] for r in regions]
        unique_colors = set(colors)
        assert len(unique_colors) <= 3, (
            f"Expected max 3 colors, got {len(unique_colors)}: {unique_colors}"
        )
        assert any(c in ["red"] for c in unique_colors), "Missing red (Bug) color"
        assert any(c in ["lime", "green"] for c in unique_colors), (
            "Missing green (Story) color"
        )
        assert any(c in ["cyan"] for c in unique_colors), "Missing cyan (Task) color"

        # Verify text patterns (should be like "Story 1", "Bug 2", "Task 3")
        texts = [r["text"] for r in regions]
        story_count = sum(1 for t in texts if "Story" in t or "story" in t)
        bug_count = sum(1 for t in texts if "Bug" in t or "bug" in t)
        task_count = sum(1 for t in texts if "Task" in t or "task" in t)

        print(f"\n  Stories: {story_count}, Bugs: {bug_count}, Tasks: {task_count}")

        # Based on analysis results: exact counts
        assert story_count == 4, f"Expected 4 stories, found {story_count}"
        assert bug_count == 5, f"Expected 5 bugs, found {bug_count}"
        assert task_count == 5, f"Expected 5 tasks, found {task_count}"

    def test_single_color_stories(self, ocr_service):
        """Test single-color scenario (all stories, same color)."""
        image_path = "test_images/stories_sample.png"

        # Verify image exists
        image = cv2.imread(image_path)
        assert image is not None, f"Could not load image: {image_path}"

        regions = ocr_service.process_image(image_path)

        # Expected: 14 stickies (all stories, based on analysis)
        assert len(regions) == 14, (
            f"Expected 14 stickies in single-color image, detected {len(regions)}"
        )

        # Verify all regions have text
        for region in regions:
            assert region["text"], "Region missing text"
            # Allow lower confidence threshold (70% min observed in analysis)
            assert region["confidence"] > 65, (
                f"Very low confidence: {region['confidence']:.1f}% for '{region['text']}'"
            )
            # Text should be reasonable length - this image has number words (3-8 chars)
            assert len(region["text"]) >= 3, (
                f"Text too short: '{region['text']}' ({len(region['text'])} chars)"
            )

        # Verify color consistency (should be predominantly 1-2 colors max)
        colors = [r["color_name"] for r in regions]
        unique_colors = set(colors)
        assert len(unique_colors) <= 2, (
            f"Expected max 2 colors in single-color image, got {len(unique_colors)}: {unique_colors}"
        )

        # Average confidence should be high
        avg_conf = sum(r["confidence"] for r in regions) / len(regions)
        print(f"\n  Average confidence: {avg_conf:.1f}%")
        assert avg_conf > 90, f"Average confidence too low: {avg_conf:.1f}%"

    def test_all_images_100_percent_detection(self, ocr_service):
        """Integration test: Verify 100% detection across all test images."""
        test_cases = [
            ("test_images/sticky_notes_sample.png", 16, "Multi-color sample"),
            ("test_images/story_bug_task_sample.png", 14, "Story/Bug/Task workflow"),
            ("test_images/stories_sample.png", 14, "Single-color stories"),
        ]

        results = []

        for image_path, expected_count, description in test_cases:
            image = cv2.imread(image_path)
            assert image is not None, f"Could not load image: {image_path}"

            regions = ocr_service.process_image(image_path)
            detected_count = len(regions)

            # Calculate stats
            confidences = [r["confidence"] for r in regions]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            min_conf = min(confidences) if confidences else 0

            texts = [r["text"] for r in regions]
            text_lengths = [len(t) for t in texts]
            avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

            results.append(
                {
                    "image": Path(image_path).name,
                    "description": description,
                    "expected": expected_count,
                    "detected": detected_count,
                    "match": detected_count == expected_count,
                    "avg_conf": avg_conf,
                    "min_conf": min_conf,
                    "avg_length": avg_length,
                }
            )

            # Assert 100% detection
            assert detected_count == expected_count, (
                f"{description}: Expected {expected_count}, got {detected_count}"
            )

        # Print summary
        print("\n" + "=" * 80)
        print("100% DETECTION VALIDATION SUMMARY")
        print("=" * 80)
        for r in results:
            status = "PASS" if r["match"] else "FAIL"
            print(
                f"{status:4s} | {r['description']:30s} | {r['detected']}/{r['expected']} stickies | "
                f"Conf: {r['avg_conf']:.1f}% avg, {r['min_conf']:.1f}% min"
            )
        print("=" * 80 + "\n")

        # All must pass
        assert all(r["match"] for r in results), (
            "Not all images achieved 100% detection"
        )

    def test_realistic_sample_production_quality(self, ocr_service, debug_output_dir):
        """
        Test production-quality OCR on realistic sticky note image.

        Validates:
        - 100% detection rate (14/14 stickies)
        - Color accuracy (lime, cyan, red)
        - Text accuracy with post-processing cleanup
        - No region splitting (multi-line text in single region)
        - High confidence (>87%)
        """
        image_path = "test_images/realistic_sample.png"

        # Expected ground truth
        EXPECTED_STICKIES = 14
        EXPECTED_COLORS = {"lime", "cyan", "red"}  # Or 'green', 'blue' variations
        MIN_CONFIDENCE = 85.0  # Minimum acceptable confidence

        # Expected text samples (key phrases to verify text accuracy)
        TEXT_ACCURACY_CHECKS = {
            "able to": True,  # Should NOT be "ableto"
            "install": True,  # Should NOT be "instal"
            "a bug": True,  # Should NOT be "bu." or "a a bug"
            "a a ": False,  # Should NOT have duplicate "a a"
            "ableto": False,  # OCR error should be fixed
            "instal ": False,  # Truncation should be fixed
        }

        # Process image
        regions = ocr_service.process_image(image_path)

        # Validate detection count
        assert len(regions) == EXPECTED_STICKIES, (
            f"Expected {EXPECTED_STICKIES} stickies, detected {len(regions)}"
        )

        # Validate colors detected
        detected_colors = {r["color_name"] for r in regions}
        # Allow variations: lime/green, cyan/blue
        color_match = (
            detected_colors == EXPECTED_COLORS
            or detected_colors == {"green", "cyan", "red"}
            or detected_colors == {"lime", "blue", "red"}
            or detected_colors == {"green", "blue", "red"}
        )
        assert color_match, (
            f"Expected colors {EXPECTED_COLORS} (or variations), got {detected_colors}"
        )

        # Validate confidence levels
        confidences = [r["confidence"] for r in regions]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        assert avg_conf > 90.0, f"Average confidence too low: {avg_conf:.1f}%"
        assert min_conf > MIN_CONFIDENCE, f"Minimum confidence too low: {min_conf:.1f}%"

        # Validate text accuracy (post-processing cleanup)
        all_text = " ".join(r["text"] for r in regions)

        for phrase, should_exist in TEXT_ACCURACY_CHECKS.items():
            if should_exist:
                assert phrase in all_text, (
                    f"Expected text '{phrase}' not found (post-processing failed)"
                )
            else:
                assert phrase not in all_text, (
                    f"OCR error '{phrase}' still present (cleanup failed)"
                )

        # Validate no region splitting (all regions should have substantial text)
        short_texts = [r for r in regions if len(r["text"].strip()) < 10]
        assert len(short_texts) == 0, (
            f"Found {len(short_texts)} regions with suspiciously short text (possible splits): "
            f"{[r['text'] for r in short_texts]}"
        )

        print("\n✅ Production Quality OCR Test PASSED:")
        print(f"   - Detection: {len(regions)}/{EXPECTED_STICKIES} (100%)")
        print(f"   - Colors: {detected_colors}")
        print(f"   - Confidence: {avg_conf:.1f}% avg, {min_conf:.1f}% min")
        print("   - Text accuracy: All checks passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
