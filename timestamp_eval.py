import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    return float(value)


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def parse_blocks(text: str) -> List[List[Dict]]:
    blocks = []
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    line_re = re.compile(
        r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
        r"(?:(\d{1,2}:\d{2}(?::\d{2})?)\s*)?[-–—]?\s*(.*?)\s*$"
    )

    for raw_block in raw_blocks:
        entries = []

        for line in raw_block.splitlines():
            line = line.strip()

            if not line:
                continue

            match = line_re.match(line)

            if not match:
                continue

            start_raw, end_raw, label = match.groups()
            entries.append({
                "start": parse_timestamp(start_raw),
                "end": parse_timestamp(end_raw) if end_raw else None,
                "label": label.strip(),
                "source": line,
            })

        if entries:
            blocks.append(entries)

    return blocks


def block_to_intervals(block: List[Dict], fallback_duration: Optional[float] = None) -> List[Dict]:
    intervals = []

    for index, item in enumerate(block):
        start = float(item["start"])
        end = item.get("end")

        if end is None:
            if index + 1 < len(block):
                end = float(block[index + 1]["start"])
            elif fallback_duration is not None:
                end = float(fallback_duration)
            else:
                continue

        end = float(end)

        if end <= start:
            continue

        intervals.append({
            "start": start,
            "end": end,
            "label": item.get("label", ""),
            "source": item.get("source", ""),
        })

    return intervals


def interval_iou(a: Dict, b: Dict) -> float:
    intersection = max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))
    union = max(float(a["end"]), float(b["end"])) - min(float(a["start"]), float(b["start"]))
    return intersection / union if union > 0 else 0.0


def evaluate_segments(reference: List[Dict], prediction: List[Dict], iou_threshold: float = 0.5) -> Dict:
    pairs: List[Tuple[float, int, int]] = []

    for pred_index, pred in enumerate(prediction):
        for ref_index, ref in enumerate(reference):
            iou = interval_iou(pred, ref)

            if iou >= iou_threshold:
                pairs.append((iou, pred_index, ref_index))

    pairs.sort(reverse=True)

    used_predictions = set()
    used_references = set()
    matches = []

    for iou, pred_index, ref_index in pairs:
        if pred_index in used_predictions or ref_index in used_references:
            continue

        used_predictions.add(pred_index)
        used_references.add(ref_index)
        matches.append({
            "iou": iou,
            "prediction": prediction[pred_index],
            "reference": reference[ref_index],
        })

    tp = len(matches)
    fp = len(prediction) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def infer_duration(blocks: List[List[Dict]]) -> Optional[float]:
    ends = [float(item["end"]) for block in blocks for item in block if item.get("end") is not None]
    return max(ends) if ends else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate timestamp segmentation with segment-level F1.")
    parser.add_argument("file", help="Path to timestamps.txt")
    parser.add_argument("--reference-block", type=int, default=0, help="Block index used as reference, zero-based")
    parser.add_argument("--prediction-block", type=int, default=2, help="Block index used as prediction, zero-based")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP match")
    parser.add_argument("--duration", default=None, help="Fallback duration for marker-only last segment, e.g. 20:00")
    args = parser.parse_args()

    blocks = parse_blocks(Path(args.file).read_text(encoding="utf-8"))

    if args.reference_block >= len(blocks) or args.prediction_block >= len(blocks):
        raise SystemExit(f"Found {len(blocks)} blocks, requested reference={args.reference_block}, prediction={args.prediction_block}")

    duration = parse_timestamp(args.duration) if args.duration else infer_duration(blocks)
    reference = block_to_intervals(blocks[args.reference_block], fallback_duration=duration)
    prediction = block_to_intervals(blocks[args.prediction_block], fallback_duration=duration)
    result = evaluate_segments(reference, prediction, iou_threshold=args.iou)

    print(f"Reference segments: {len(reference)}")
    print(f"Prediction segments: {len(prediction)}")
    print(f"IoU threshold: {args.iou:.2f}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    print(f"Precision={result['precision']:.4f}")
    print(f"Recall={result['recall']:.4f}")
    print(f"F1={result['f1']:.4f}")

    print("\nMatches:")
    for match in result["matches"]:
        pred = match["prediction"]
        ref = match["reference"]
        print(
            f"IoU={match['iou']:.3f} | "
            f"pred {format_timestamp(pred['start'])}-{format_timestamp(pred['end'])} {pred['label']} | "
            f"ref {format_timestamp(ref['start'])}-{format_timestamp(ref['end'])} {ref['label']}"
        )


if __name__ == "__main__":
    main()
