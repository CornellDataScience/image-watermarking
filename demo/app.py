import os
import sys
import tempfile
import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Option 3 (centroid matching)
from watermark.encoder import encode_watermark, EncodeParams
from watermark.decoder import decode_watermark, DecodeError
from watermark.sidecar import sidecar_to_file, sidecar_from_file
from watermark.reed_solomon import ERASURE, bits_to_bytes_with_erasures
from watermark.centroid_matching import compute_region_centroids, match_regions_by_centroid

# Option 2 (IoU+Hungarian matching)
from watermark.encoder_b import encode_watermark_b, EncodeParamsB
from watermark.decoder_b import decode_watermark_b, DecodeErrorB
from watermark.sidecar_b import sidecar_b_to_file, sidecar_b_from_file, SIDECAR_B_MAGIC

from regions.approach_regions import slic_superpixels
from descriptors.dwt_descriptor import compute_raw_dwt_ll
from core.types import SegmentationResult
from stability.region_matching import match_regions as iou_match_regions


def _detect_method(sidecar_path: str) -> str:
    with open(sidecar_path, "rb") as f:
        magic = f.read(4)
    return "option2" if magic == SIDECAR_B_MAGIC else "option3"


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def run_encode(image_path, message, k, rs_overhead, key, method):
    if image_path is None:
        return None, "Upload an image first."
    if not message.strip():
        return None, "Enter a message to encode."

    image = cv2.imread(image_path)
    if image is None:
        return None, f"Could not read image: {image_path}"

    try:
        key_int = int(key)
    except ValueError:
        return None, "Key must be an integer."

    msg_bytes = message.encode("utf-8")
    if len(msg_bytes) > 127:
        return None, f"Message too long ({len(msg_bytes)} bytes). Max is 127 bytes at 2× RS overhead."

    params3 = EncodeParams(k=int(k), rs_overhead=float(rs_overhead), min_margin=0.05)
    params2 = EncodeParamsB(k=int(k), rs_overhead=float(rs_overhead), min_margin=0.05, iou_threshold=0.5)

    tmp = tempfile.NamedTemporaryFile(suffix=".wm", delete=False)
    tmp.close()

    try:
        if method == "Option 2 — IoU matching (larger sidecar, more accurate)":
            sidecar = encode_watermark_b(image, msg_bytes, key_int, params2)
            sidecar_b_to_file(sidecar, tmp.name)
            n_regions = sidecar.num_regions
            n_pairs   = len(sidecar.pairs)
            method_label = "Option 2 (IoU + Hungarian)"
        else:
            sidecar = encode_watermark(image, msg_bytes, key_int, params3)
            sidecar_to_file(sidecar, tmp.name)
            n_regions = len(sidecar.centroids)
            n_pairs   = len(sidecar.pairs)
            method_label = "Option 3 (Centroid matching)"
    except ValueError as e:
        return None, f"Encode error: {e}"

    size_kb = os.path.getsize(tmp.name) / 1024
    summary = (
        f"Encoded successfully.\n\n"
        f"  Method:           {method_label}\n"
        f"  Message:          \"{message}\"\n"
        f"  Message length:   {len(msg_bytes)} bytes\n"
        f"  k:                {int(k)} witnesses/bit\n"
        f"  RS overhead:      {float(rs_overhead):.1f}×\n"
        f"  RS-encoded bits:  {n_pairs}\n"
        f"  Regions used:     {n_regions}\n"
        f"  Sidecar size:     {size_kb:.1f} KB\n"
        f"  Image dimensions: {image.shape[1]}×{image.shape[0]} px\n\n"
        f"Download the sidecar file and keep it — you'll need it to decode."
    )
    return tmp.name, summary


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def run_decode(altered_image_path, sidecar_file, original_image_path):
    if altered_image_path is None:
        return "Upload the altered image."
    if sidecar_file is None:
        return "Upload the sidecar (.wm) file."

    sidecar_path = sidecar_file.name if hasattr(sidecar_file, 'name') else sidecar_file

    after = cv2.imread(altered_image_path)
    if after is None:
        return "Could not read altered image."

    detected = _detect_method(sidecar_path)

    if detected == "option2":
        return _decode_option2(after, sidecar_path, original_image_path)
    else:
        return _decode_option3(after, sidecar_path, original_image_path)


def _decode_option2(after, sidecar_path, original_image_path):
    try:
        sidecar = sidecar_b_from_file(sidecar_path)
    except Exception as e:
        return f"Could not read Option 2 sidecar: {e}"

    rs_success = False
    rs_result  = None
    try:
        recovered = decode_watermark_b(after, sidecar)
        rs_result = recovered.decode("utf-8", errors="replace")
        rs_success = True
    except DecodeErrorB:
        pass

    meta        = sidecar.metadata
    msg_len     = meta.get("message_length", None)
    rs_overhead = meta.get("rs_overhead", 2.0)
    k           = meta.get("k", 7)
    iou_thresh  = meta.get("iou_threshold", 0.5)
    before_h    = meta.get("image_height")
    before_w    = meta.get("image_width")

    if before_h and before_w:
        ah, aw = after.shape[:2]
        after_r = cv2.resize(after, (before_w, before_h), interpolation=cv2.INTER_LINEAR) \
                  if (ah, aw) != (before_h, before_w) else after
    else:
        after_r = after

    seg_a = slic_superpixels(after_r)
    d_a   = compute_raw_dwt_ll(after_r, seg_a)

    seg_before = SegmentationResult(
        region_map=sidecar.region_map.astype(int),
        num_regions=sidecar.num_regions,
    )
    match = iou_match_regions(seg_before, seg_a, iou_threshold=iou_thresh)

    matched_count = sum(1 for v in match.values() if v is not None)
    total_regions = len(match)

    raw_output = None
    flip_rate = byte_err = rs_cap = None
    correct = wrong = erased_bits = 0

    # Option 2 stores descriptors in the sidecar — no need for original image
    d_o = sidecar.descriptors
    voted_bits = []
    for pairs_j in sidecar.pairs:
        votes = []
        for (r1, r2) in pairs_j:
            s1, s2 = match.get(r1), match.get(r2)
            if s1 is None or s2 is None:
                continue
            votes.append(1 if d_a[s1] > d_a[s2] else 0)
        if not votes:
            erased_bits += 1
            voted_bits.append(ERASURE)
        else:
            voted_bits.append(1 if sum(votes) > len(votes) / 2 else 0)

    for j, (pairs_j, vb) in enumerate(zip(sidecar.pairs, voted_bits)):
        if vb == ERASURE:
            continue
        r1, r2 = pairs_j[0]
        exp = 1 if d_o[r1] > d_o[r2] else 0
        if vb == exp:
            correct += 1
        else:
            wrong += 1

    non_erased = correct + wrong
    flip_rate  = wrong / non_erased if non_erased else 0.0
    byte_err   = 1 - (1 - flip_rate) ** 8
    rs_cap     = ((msg_len * (rs_overhead - 1)) // 2) / (msg_len * rs_overhead) if msg_len else None

    raw_bytes, _ = bits_to_bytes_with_erasures(voted_bits)
    if msg_len:
        raw_output = raw_bytes[:msg_len].decode("utf-8", errors="replace")

    lines = []
    if rs_success:
        lines.append("DECODE SUCCESS  [Option 2 — IoU matching]")
        lines.append(f"\n  Recovered message: \"{rs_result}\"")
    else:
        lines.append("DECODE FAILED — edit was too destructive for RS to recover.  [Option 2 — IoU matching]")
        if raw_output is not None:
            lines.append(f"\n  Raw majority-vote output (pre-RS): \"{raw_output}\"")
            lines.append("  (This is garbled — shown so you can see how much signal survived)")

    lines.append(f"\nDiagnostics:")
    lines.append(f"  Image dimensions:  {after.shape[1]}×{after.shape[0]} px  (original: {before_w}×{before_h})")
    lines.append(f"  k: {k}  |  RS overhead: {rs_overhead:.1f}×")
    lines.append(f"  Regions matched:   {matched_count}/{total_regions} ({100*matched_count/total_regions:.1f}%)  [IoU ≥ {iou_thresh}]")

    if flip_rate is not None:
        lines.append(f"  Bit flip rate:     {flip_rate*100:.2f}%  ({wrong} wrong / {non_erased} non-erased bits)")
        lines.append(f"  Byte error rate:   {byte_err*100:.1f}%")
        if rs_cap is not None:
            lines.append(f"  RS {rs_overhead:.1f}x capacity:    {rs_cap*100:.1f}%  ({'OK' if byte_err < rs_cap else 'EXCEEDED'})")
    else:
        lines.append("  (Upload original image to see bit-level diagnostics)")

    return "\n".join(lines)


def _decode_option3(after, sidecar_path, original_image_path):
    try:
        sidecar = sidecar_from_file(sidecar_path)
    except Exception as e:
        return f"Could not read Option 3 sidecar: {e}"

    rs_success = False
    rs_result  = None
    try:
        recovered = decode_watermark(after, sidecar)
        rs_result = recovered.decode("utf-8", errors="replace")
        rs_success = True
    except DecodeError:
        pass

    meta        = sidecar.metadata
    msg_len     = meta.get("message_length", None)
    rs_overhead = meta.get("rs_overhead", 2.0)
    k           = meta.get("k", 7)
    before_h    = meta.get("image_height")
    before_w    = meta.get("image_width")
    ct          = meta.get("centroid_threshold", 40.0)

    if before_h and before_w:
        ah, aw = after.shape[:2]
        after_r = cv2.resize(after, (before_w, before_h), interpolation=cv2.INTER_LINEAR) \
                  if (ah, aw) != (before_h, before_w) else after
    else:
        after_r = after

    seg_a = slic_superpixels(after_r)
    d_a   = compute_raw_dwt_ll(after_r, seg_a)
    c_a   = compute_region_centroids(seg_a)

    match          = match_regions_by_centroid(sidecar.centroids, c_a, centroid_threshold=ct)
    erased_regions = sum(1 for v in match.values() if v is None)
    total_regions  = len(match)

    raw_output = None
    flip_rate = byte_err = rs_cap = None
    correct = wrong = erased_bits = 0

    if original_image_path is not None:
        orig = cv2.imread(original_image_path)
        if orig is not None:
            seg_o = slic_superpixels(orig)
            d_o   = compute_raw_dwt_ll(orig, seg_o)

            voted_bits = []
            for pairs_j in sidecar.pairs:
                votes = []
                for (r1, r2) in pairs_j:
                    s1, s2 = match.get(r1), match.get(r2)
                    if s1 is None or s2 is None:
                        continue
                    votes.append(1 if d_a[s1] > d_a[s2] else 0)
                if not votes:
                    erased_bits += 1
                    voted_bits.append(ERASURE)
                else:
                    voted_bits.append(1 if sum(votes) > len(votes) / 2 else 0)

            for j, (pairs_j, vb) in enumerate(zip(sidecar.pairs, voted_bits)):
                if vb == ERASURE:
                    continue
                r1, r2 = pairs_j[0]
                exp = 1 if d_o[r1] > d_o[r2] else 0
                if vb == exp:
                    correct += 1
                else:
                    wrong += 1

            non_erased = correct + wrong
            flip_rate  = wrong / non_erased if non_erased else 0.0
            byte_err   = 1 - (1 - flip_rate) ** 8
            rs_cap     = ((msg_len * (rs_overhead - 1)) // 2) / (msg_len * rs_overhead) if msg_len else None

            raw_bytes, _ = bits_to_bytes_with_erasures(voted_bits)
            if msg_len:
                raw_output = raw_bytes[:msg_len].decode("utf-8", errors="replace")

    lines = []
    if rs_success:
        lines.append("DECODE SUCCESS  [Option 3 — Centroid matching]")
        lines.append(f"\n  Recovered message: \"{rs_result}\"")
    else:
        lines.append("DECODE FAILED — edit was too destructive for RS to recover.  [Option 3 — Centroid matching]")
        if raw_output is not None:
            lines.append(f"\n  Raw majority-vote output (pre-RS): \"{raw_output}\"")
            lines.append("  (This is garbled — shown so you can see how much signal survived)")

    lines.append(f"\nDiagnostics:")
    lines.append(f"  Image dimensions:  {after.shape[1]}×{after.shape[0]} px  (original: {before_w}×{before_h})")
    lines.append(f"  k: {k}  |  RS overhead: {rs_overhead:.1f}×")
    lines.append(
        f"  Regions found:     {total_regions - erased_regions}/{total_regions} "
        f"({100*(total_regions-erased_regions)/total_regions:.1f}%)  [centroid drift ≤ {ct:.0f}px]"
    )

    if flip_rate is not None:
        lines.append(f"  Bit flip rate:     {flip_rate*100:.2f}%  ({wrong} wrong / {non_erased} non-erased bits)")
        lines.append(f"  Byte error rate:   {byte_err*100:.1f}%")
        if rs_cap is not None:
            lines.append(f"  RS {rs_overhead:.1f}x capacity:    {rs_cap*100:.1f}%  ({'OK' if byte_err < rs_cap else 'EXCEEDED'})")
    else:
        lines.append("  (Upload original image to see bit-level diagnostics)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

METHOD_CHOICES = [
    "Option 2 — IoU matching (larger sidecar, more accurate)",
    "Option 3 — Centroid matching (compact sidecar, faster)",
]

with gr.Blocks(title="Image Watermarking Demo") as demo:
    gr.Markdown(
        "# Image Watermarking Demo\n"
        "Embed a message into an image by recording pairwise region orderings. "
        "No pixels are modified.\n\n"
        "**Option 2** stores the full segmentation map in the sidecar and uses "
        "IoU + Hungarian matching at decode time (more accurate, larger file).\n"
        "**Option 3** stores only region centroids (compact sidecar, faster decode)."
    )

    with gr.Tab("Encode"):
        gr.Markdown(
            "Upload the **original image**, choose a method, type a message, and "
            "download the sidecar file. Keep the sidecar — you'll need it to decode."
        )
        with gr.Row():
            with gr.Column():
                enc_image   = gr.Image(label="Original image", type="filepath")
                enc_method  = gr.Radio(
                    choices=METHOD_CHOICES,
                    value=METHOD_CHOICES[0],
                    label="Encoding method",
                )
                enc_message = gr.Textbox(label="Message to encode", placeholder="hi im a dog named bruno")
                enc_k       = gr.Slider(
                    minimum=3, maximum=21, step=2, value=7,
                    label="k — witnesses per bit",
                    info="How many region pairs vote on each bit. Higher = more robust against image edits, but needs more regions. Try 7 for light edits, 11–15 for heavy ones.",
                )
                enc_rs      = gr.Slider(
                    minimum=1.5, maximum=4.0, step=0.5, value=2.0,
                    label="RS overhead — redundancy multiplier",
                    info="How many times the payload is expanded for error correction. 2× can fix ~25% bit errors; 3× can fix ~33%. Higher = more resilient, but fewer messages fit.",
                )
                with gr.Accordion("Advanced", open=False):
                    enc_key = gr.Textbox(label="Watermark ID", value="42", info="Seeds the pair assignment — change this if you need two watermarks on the same image to not interfere.")
                enc_btn = gr.Button("Encode", variant="primary")
            with gr.Column():
                enc_sidecar = gr.File(label="Download sidecar (.wm)")
                enc_summary = gr.Textbox(label="Summary", lines=12, interactive=False)

        enc_btn.click(
            fn=run_encode,
            inputs=[enc_image, enc_message, enc_k, enc_rs, enc_key, enc_method],
            outputs=[enc_sidecar, enc_summary],
        )

    with gr.Tab("Decode"):
        gr.Markdown(
            "Upload the **altered image** and the **sidecar (.wm)** from the encode step. "
            "The method (Option 2 or Option 3) is detected automatically from the sidecar. "
            "Optionally upload the original image for full diagnostics."
        )
        with gr.Row():
            with gr.Column():
                dec_altered  = gr.Image(label="Altered image", type="filepath")
                dec_sidecar  = gr.File(label="Sidecar (.wm)", file_types=[".wm"])
                dec_original = gr.Image(
                    label="Original image (optional — for diagnostics)",
                    type="filepath",
                )
                dec_btn = gr.Button("Decode", variant="primary")
            with gr.Column():
                dec_result = gr.Textbox(label="Result", lines=18, interactive=False)

        dec_btn.click(
            fn=run_decode,
            inputs=[dec_altered, dec_sidecar, dec_original],
            outputs=[dec_result],
        )


if __name__ == "__main__":
    demo.launch()
