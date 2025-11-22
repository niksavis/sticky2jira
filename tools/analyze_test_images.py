"""
Analyze test images to find optimal OCR parameters.

WHEN TO USE THIS TOOL:
----------------------
1. **Testing new sticky note images** - Validates if current OCR settings work on your images
2. **Troubleshooting detection issues** - Compares different parameter combinations to find what works best
3. **Before deploying to new teams** - Ensures OCR works with their sticky note colors/styles

HOW TO USE:
-----------
1. Place your test images in test_images/ folder
2. Update the image list at bottom of this file with expected sticky counts
3. Run: python analyze_test_images.py
4. Review output to see which configuration achieves 100% detection
5. If needed, update defaults in services/ocr_service.py

WHAT IT TESTS:
--------------
- Multiple HSV tolerance values (color detection sensitivity)
- Different minimum size thresholds (small vs large stickies)
- Various proximity settings (how close stickies can be)

OUTPUT:
-------
- Console: Detailed results for each configuration
- test_output/ocr_parameter_analysis.json: Machine-readable results for comparison
"""

import cv2
from pathlib import Path
from services.ocr_service import OCRService
import json
from typing import Optional


def analyze_image(image_path: str, expected_count: Optional[int] = None):
    """
    Test OCR on a single image with multiple parameter configurations.

    Args:
        image_path: Path to test image
        expected_count: Expected number of stickies (None = just report what's detected)

    Returns:
        List of results for each configuration tested
    """
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {image_path}")
    print(f"{'=' * 80}")

    # Load image to get basic stats
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image from {image_path}")
        return []

    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")
    if expected_count:
        print(f"Expected stickies: {expected_count}")
    else:
        print("Expected stickies: UNKNOWN (will report detected count)")

    # CURRENT PRODUCTION SETTINGS (baseline)
    configs = [
        {
            "name": "CURRENT PRODUCTION (100px clustering)",
            "hsv_tolerance": 20,
            "min_size": 1500,
            "max_size": 50000,
            "proximity": 100,  # Current optimal value
        },
        {
            "name": "Alternative: Wider Color Tolerance",
            "hsv_tolerance": 30,
            "min_size": 1500,
            "max_size": 50000,
            "proximity": 100,
        },
        {
            "name": "Alternative: Smaller Stickies Allowed",
            "hsv_tolerance": 20,
            "min_size": 1000,  # Lower threshold
            "max_size": 50000,
            "proximity": 100,
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'-' * 80}")
        print(f"Testing: {config['name']}")
        print(
            f"  Parameters: hsv_tolerance={config['hsv_tolerance']}, "
            f"min_size={config['min_size']}, proximity={config['proximity']}"
        )

        try:
            ocr_service = OCRService(
                hsv_tolerance=config["hsv_tolerance"],
                min_size=config["min_size"],
                max_size=config["max_size"],
                proximity=config["proximity"],
            )

            regions = ocr_service.process_image(image_path)

            # Analyze results
            detected_count = len(regions)
            confidences = [r["confidence"] for r in regions]
            colors = [r["color_name"] for r in regions]
            text_lengths = [len(r["text"]) for r in regions]

            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            min_conf = min(confidences) if confidences else 0
            avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            short_texts = sum(1 for t in text_lengths if t < 10)

            # Verdict
            if expected_count and detected_count == expected_count:
                status = "PASS"
            elif not expected_count:
                status = "INFO"
            else:
                status = "FAIL"

            print(f"  [{status}] Detected: {detected_count} stickies")
            print(f"     Confidence: {avg_conf:.1f}% avg (min: {min_conf:.1f}%)")
            print(
                f"     Text quality: {avg_length:.1f} chars avg, {short_texts} short texts"
            )
            print(f"     Colors: {sorted(set(colors))}")

            # Show sample texts (first 5)
            print("\n  Sample detected texts:")
            for i, r in enumerate(regions[:5]):
                quality = (
                    "OK" if len(r["text"]) >= 10 and r["confidence"] > 85 else "WARNING"
                )
                print(
                    f"    [{quality:7s}] [{r['color_name']:8s}] {r['confidence']:5.1f}% | {r['text'][:50]}"
                )

            if len(regions) > 5:
                print(f"    ... and {len(regions) - 5} more")

            results.append(
                {
                    "config": config["name"],
                    "params": {k: v for k, v in config.items() if k != "name"},
                    "detected_count": detected_count,
                    "avg_confidence": avg_conf,
                    "min_confidence": min_conf,
                    "avg_text_length": avg_length,
                    "short_text_count": short_texts,
                    "colors": sorted(set(colors)),
                    "matches_expected": detected_count == expected_count
                    if expected_count
                    else None,
                }
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {Path(image_path).name}")
    print(f"{'=' * 80}")

    if expected_count:
        matching = [r for r in results if r["matches_expected"]]
        if matching:
            print(
                f"\n[SUCCESS] {len(matching)} configuration(s) achieved 100% detection:"
            )
            for r in matching:
                print(f"   - {r['config']}: {r['avg_confidence']:.1f}% avg confidence")
        else:
            print("\n[FAILED] NO configurations matched expected count!")
            print("\nClosest results:")
            sorted_results = sorted(
                results, key=lambda x: abs(x["detected_count"] - expected_count)
            )
            for r in sorted_results[:2]:
                diff = r["detected_count"] - expected_count
                print(
                    f"   - {r['config']}: {r['detected_count']} ({diff:+d}), {r['avg_confidence']:.1f}% conf"
                )
    else:
        print("\n[INFO] Detection counts (expected count not provided):")
        for r in results:
            print(
                f"   - {r['config']}: {r['detected_count']} stickies, {r['avg_confidence']:.1f}% conf"
            )

    return results


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    OCR PARAMETER VALIDATION TOOL                           ║
║                                                                            ║
║  Purpose: Test if current OCR settings work on your sticky note images    ║
║  Output:  Detailed comparison of different parameter configurations       ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Configure your test images here
    test_cases = [
        {
            "path": "test_images/sticky_notes_sample.png",
            "expected": 16,
            "description": "Multi-color test (11 colors)",
        },
        {
            "path": "test_images/story_bug_task_sample.png",
            "expected": 14,
            "description": "3-type workflow (Story/Bug/Task)",
        },
        {
            "path": "test_images/stories_sample.png",
            "expected": 14,
            "description": "Single-type workflow (Stories only)",
        },
        {
            "path": "test_images/realistic_sample.png",
            "expected": 14,
            "description": "Production quality test",
        },
    ]

    all_results = {}

    for test_case in test_cases:
        if not Path(test_case["path"]).exists():
            print(f"\n[WARNING] SKIPPED: {test_case['path']} (file not found)")
            continue

        print(f"\n\n{'=' * 80}")
        print(f"TEST: {test_case['description']}")
        print(f"{'=' * 80}")

        image_name = Path(test_case["path"]).stem
        all_results[image_name] = analyze_image(
            test_case["path"], expected_count=test_case["expected"]
        )

    # Save results
    output_file = "test_output/ocr_parameter_analysis.json"
    Path("test_output").mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 80}")
    print(f"Analysis complete. Results saved to: {output_file}")
    print(f"{'=' * 80}\n")

    # Final recommendations
    print("""
INTERPRETATION GUIDE:
────────────────────
[PASS]    = Configuration matches expected count (100% detection)
[FAIL]    = Mismatch (under/over detection)
[INFO]    = No expected count provided (informational only)

[WARNING] = Low confidence or short text (may need manual review)
[OK]      = Good quality detection

WHAT TO DO NEXT:
────────────────
1. If CURRENT PRODUCTION shows ✅ for all images → OCR is working perfectly
2. If any ❌ appear → Check which alternative configuration works better
3. If alternatives fail too → Image may need preprocessing or manual region drawing
4. Update defaults in services/ocr_service.py if better parameters found

TROUBLESHOOTING:
────────────────
- Under-detection: Try "Smaller Stickies Allowed" or "Wider Color Tolerance"
- Over-detection: Increase min_size to filter out noise
- Wrong colors: Check HSV values with debug tools (contact developer)
    """)
