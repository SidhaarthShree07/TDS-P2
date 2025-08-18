import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    import pytesseract
except Exception as e:  # pragma: no cover
    pytesseract = None  # type: ignore


def _ensure_tesseract_cmd() -> None:
    """
    Try to locate the `tesseract` binary on Windows/Linux/Mac and set
    pytesseract.pytesseract.tesseract_cmd if needed.
    Users can also set the TESSERACT_CMD env var explicitly.
    """
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Install it and the Tesseract binary.")

    # Respect explicit override first
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    # Common Windows locations
    win_paths = [
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Users\\Public\\Tesseract-OCR\\tesseract.exe",
    ]
    for p in win_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return
    # Otherwise rely on PATH lookup; if missing, pytesseract will raise later


def _load_and_preprocess(image_path: str) -> Image.Image:
    """
    Generic, conservative preprocessing that tends to help OCR across a wide
    range of document styles without overfitting to tables.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to grayscale and apply adaptive contrast
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    # Light sharpen to improve character edges
    g = g.filter(ImageFilter.SHARPEN)

    # If background is very dark (e.g., white text on navy bar), keep a copy of native RGB for later
    return g


def _ocr_full_text(image: Image.Image, lang: str = "eng") -> str:
    custom_config = "--oem 3 --psm 6"  # OEM LSTM, assume a block of text
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        return text.strip()
    except Exception as e:
        return ""


def _ocr_words_dataframe(image: Image.Image, lang: str = "eng") -> List[Dict[str, Any]]:
    """
    Return the detailed word-level data from Tesseract (similar to TSV as dicts).
    Each item has at least: level, left, top, width, height, conf, text, block_num, par_num, line_num, word_num
    """
    config = "--oem 3 --psm 6"
    try:
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    n = len(data.get("text", []))
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        txt = data["text"][i]
        if txt is None:
            txt = ""
        rows.append({
            "text": str(txt).strip(),
            "conf": float(data.get("conf", ["-1"]) [i]) if str(data.get("conf", ["-1"]) [i]).replace('.', '', 1).lstrip('-').isdigit() else -1.0,
            "left": int(data.get("left", [0])[i]),
            "top": int(data.get("top", [0])[i]),
            "width": int(data.get("width", [0])[i]),
            "height": int(data.get("height", [0])[i]),
            "block_num": int(data.get("block_num", [0])[i]),
            "par_num": int(data.get("par_num", [0])[i]),
            "line_num": int(data.get("line_num", [0])[i]),
            "word_num": int(data.get("word_num", [0])[i]),
            "page_num": int(data.get("page_num", [0])[i]) if "page_num" in data else 1,
        })
    return rows


def _group_lines(word_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group word rows into lines using (block_num, par_num, line_num).
    Returns a list of dicts: {"bbox": (l,t,r,b), "words": [...], "text": "..."}
    """
    lines: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for w in word_rows:
        if not w.get("text"):
            continue
        key = (w["block_num"], w["par_num"], w["line_num"])
        lines.setdefault(key, []).append(w)

    out: List[Dict[str, Any]] = []
    for key, words in lines.items():
        words = sorted(words, key=lambda d: d["left"])  # left-to-right
        text = " ".join([w["text"] for w in words if w["text"]])
        left = min(w["left"] for w in words)
        top = min(w["top"] for w in words)
        right = max(w["left"] + w["width"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)
        out.append({"bbox": (left, top, right, bottom), "words": words, "text": text})
    # Sort by top, then left
    out.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return out


def _extract_key_values(lines: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Heuristic key-value extraction:
    - lines that contain a colon, hyphen, or tab-like spacing
    - also detect two-chunk lines by large gaps between consecutive words
    """
    result: Dict[str, str] = {}
    for line in lines:
        t = line["text"].strip()
        if not t:
            continue
        # Case 1: colon separated
        if ":" in t:
            parts = [p.strip(" -:\t") for p in t.split(":", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                result[parts[0]] = parts[1]
                continue
        # Case 2: hyphen separated key - value
        if " - " in t and t.count(" - ") == 1:
            k, v = t.split(" - ", 1)
            if k.strip() and v.strip():
                result[k.strip()] = v.strip()
                continue
        # Case 3: large visual gap suggests two columns (key value)
        words = line["words"]
        if len(words) >= 2:
            gaps = []
            for a, b in zip(words, words[1:]):
                gap = b["left"] - (a["left"] + a["width"])
                gaps.append(gap)
            if gaps:
                max_gap_idx = int(np.argmax(gaps))
                if gaps[max_gap_idx] > 40:  # pixels
                    left_words = [w["text"] for w in words[: max_gap_idx + 1]]
                    right_words = [w["text"] for w in words[max_gap_idx + 1 :]]
                    k = " ".join(left_words).strip()
                    v = " ".join(right_words).strip()
                    if k and v and k not in result:
                        result[k] = v
    return result


def _cluster_columns(lines: List[Dict[str, Any]]) -> List[int]:
    """
    Determine likely column x-centers using simple binning over word-left positions
    across a set of lines. The output is a sorted list of x-centers.
    """
    xs: List[int] = []
    for line in lines:
        for w in line["words"]:
            xs.append(w["left"])
    if not xs:
        return []
    xs = sorted(xs)
    bins: List[List[int]] = []
    for x in xs:
        placed = False
        for b in bins:
            if abs(x - np.mean(b)) <= 40:  # pixels tolerance
                b.append(x)
                placed = True
                break
        if not placed:
            bins.append([x])
    centers = sorted([int(np.mean(b)) for b in bins])
    return centers


def _extract_tables(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Heuristic table detection:
    - Scan consecutive lines where each line has at least 3 words
    - Build column centers from word x-positions
    - Assign each word to nearest column and generate rows
    """
    tables: List[Dict[str, Any]] = []
    current_group: List[Dict[str, Any]] = []

    def flush_group():
        if len(current_group) < 3:
            return
        centers = _cluster_columns(current_group)
        if len(centers) < 3:
            return
        # Build matrix rows
        rows: List[List[str]] = []
        for ln in current_group:
            cols = [""] * len(centers)
            for w in ln["words"]:
                idx = int(np.argmin([abs((w["left"] + w["width"]/2) - c) for c in centers]))
                cols[idx] = (cols[idx] + " " + w["text"]).strip()
            rows.append(cols)
        # First row as header if it looks like titles (no numbers-only dominance)
        header = None
        if rows:
            header = rows[0]
            body = rows[1:]
        else:
            body = []
        tables.append({"columns": header or [], "rows": body})

    # Grouping by vertical proximity and minimum words per line
    prev_bottom = None
    for ln in lines:
        if len([w for w in ln["words"] if w["text"]]) < 3:
            if current_group:
                flush_group()
                current_group = []
            prev_bottom = ln["bbox"][3]
            continue
        if prev_bottom is None:
            current_group = [ln]
            prev_bottom = ln["bbox"][3]
            continue
        gap = ln["bbox"][1] - prev_bottom
        if gap > 35:  # a bigger vertical gap breaks the table block
            flush_group()
            current_group = [ln]
        else:
            current_group.append(ln)
        prev_bottom = ln["bbox"][3]
    if current_group:
        flush_group()

    return tables


def _paragraphs_from_lines(lines: List[Dict[str, Any]]) -> List[str]:
    paras: List[str] = []
    buf: List[str] = []
    prev_bottom = None
    for ln in lines:
        text = ln["text"].strip()
        if not text:
            continue
        if prev_bottom is None:
            buf = [text]
            prev_bottom = ln["bbox"][3]
            continue
        gap = ln["bbox"][1] - prev_bottom
        if gap > 25 and buf:
            paras.append(" ".join(buf))
            buf = [text]
        else:
            buf.append(text)
        prev_bottom = ln["bbox"][3]
    if buf:
        paras.append(" ".join(buf))
    return paras


def ocr_extract(image_path: str, lang: str = "eng") -> Dict[str, Any]:
    """
    Perform OCR on an image and return a generic structured object suitable for LLMs.

    Returns dict with keys:
      - success: bool
      - text: full OCR text
      - key_values: {key: value}
      - tables: [{columns: [...], rows: [[...], ...]}]
      - paragraphs: ["...", ...]
      - meta: {width, height}
    """
    _ensure_tesseract_cmd()
    img = _load_and_preprocess(image_path)

    full_text = _ocr_full_text(img, lang=lang)
    word_rows = _ocr_words_dataframe(img, lang=lang)
    lines = _group_lines(word_rows)

    key_values = _extract_key_values(lines)
    tables = _extract_tables(lines)
    paragraphs = _paragraphs_from_lines(lines)

    w, h = img.size
    return {
        "success": True,
        "text": full_text,
        "key_values": key_values,
        "tables": tables,
        "paragraphs": paragraphs,
        "meta": {"width": w, "height": h, "lang": lang},
    }


def ocr_extract_bytes(image_bytes: bytes, lang: str = "eng") -> Dict[str, Any]:
    """
    Same as ocr_extract, but accepts raw image bytes (useful for FastAPI uploads).
    """
    _ensure_tesseract_cmd()
    img = Image.open(np.frombuffer(image_bytes, dtype=np.uint8))
    # PIL can accept a file-like object, but frombuffer trick sometimes fails for PNG/JPEG; handle robustly
    try:
        img = Image.open(np.frombuffer(image_bytes, dtype=np.uint8))
    except Exception:
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    processed = ImageOps.autocontrast(ImageOps.grayscale(img)).filter(ImageFilter.SHARPEN)

    full_text = _ocr_full_text(processed, lang=lang)
    word_rows = _ocr_words_dataframe(processed, lang=lang)
    lines = _group_lines(word_rows)

    key_values = _extract_key_values(lines)
    tables = _extract_tables(lines)
    paragraphs = _paragraphs_from_lines(lines)

    w, h = processed.size
    return {
        "success": True,
        "text": full_text,
        "key_values": key_values,
        "tables": tables,
        "paragraphs": paragraphs,
        "meta": {"width": w, "height": h, "lang": lang},
    }


def to_readable_summary(struct: Dict[str, Any]) -> str:
    """
    Produce a compact, human-readable string summarizing OCR results.
    """
    parts: List[str] = []
    parts.append("== OCR SUMMARY ==")
    if struct.get("key_values"):
        parts.append("-- Key/Value Pairs --")
        for k, v in struct["key_values"].items():
            parts.append(f"{k}: {v}")
    if struct.get("tables"):
        for i, tbl in enumerate(struct["tables"], 1):
            parts.append(f"-- Table {i} --")
            if tbl.get("columns"):
                parts.append(" | ".join(tbl["columns"]))
            for row in tbl.get("rows", []):
                parts.append(" | ".join(str(c) for c in row))
    if struct.get("paragraphs"):
        parts.append("-- Paragraphs --")
        for p in struct["paragraphs"]:
            parts.append(p)
    return "\n".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generic OCR extractor")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    result = ocr_extract(args.image, lang=args.lang)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()
        print(to_readable_summary(result))

