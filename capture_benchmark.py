"""
Network payload benchmark for WPU face recognition.

Captures N frames from PiCamera2, applies Laplacian blur quality gate,
and saves two artefacts per accepted frame:
  - benchmark_output/full_frames/  -> full 640x480 JPEG  (Option 1 payload)
  - benchmark_output/face_crops/   -> cropped face JPEG  (Option 2a payload)

Mirrors face_service.py exactly: cam.start() only, no preview, logs every
frame decision to terminal. Press Ctrl+C to stop early — partial results saved.

Usage:
    uv run python3 capture_benchmark.py [--frames N] [--blur-threshold T] [--jpeg-quality Q]

Defaults:
    --frames          30
    --blur-threshold  120.0
    --jpeg-quality    85
"""

import argparse
import json
import logging
import os
import signal
import time

import cv2
import numpy as np
from picamera2 import Picamera2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

YUNET_MODEL_PATH = "/home/dreamvu/wpu_client/face_detection_yunet_2023mar.onnx"

OUTPUT_DIR      = "benchmark_output"
FULL_FRAMES_DIR = os.path.join(OUTPUT_DIR, "full_frames")
FACE_CROPS_DIR  = os.path.join(OUTPUT_DIR, "face_crops")
STATS_FILE      = os.path.join(OUTPUT_DIR, "size_summary.json")

MIN_FACE_SIZE = 100


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def encode_jpeg(img_bgr: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def run_benchmark(max_frames: int, blur_threshold: float, jpeg_quality: int) -> None:
    os.makedirs(FULL_FRAMES_DIR, exist_ok=True)
    os.makedirs(FACE_CROPS_DIR, exist_ok=True)

    # ── Camera — mirrors face_service.py exactly ──────────────────────────
    logger.info("Starting Picamera2 (640×480 RGB888)...")
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
    ))
    cam.start()
    logger.info("Picamera2 started successfully (640×480 RGB888)")
    time.sleep(1.0)   # let AEC/AGC settle

    # ── YuNet — mirrors face_service.py exactly ───────────────────────────
    logger.info(f"Loading YuNet detector...")
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )
    logger.info(f"YuNet detector initialised: {os.path.basename(YUNET_MODEL_PATH)}")

    # ── Stats ─────────────────────────────────────────────────────────────
    total_captured     = 0
    rejected_blur      = 0
    rejected_no_face   = 0
    rejected_too_small = 0
    accepted           = 0
    full_sizes: list[int] = []
    face_sizes: list[int] = []

    stop = {"now": False}
    original_sigint = signal.getsignal(signal.SIGINT)
    def _sigint(sig, frame):
        logger.info("Ctrl+C — stopping early, saving partial results...")
        stop["now"] = True
    signal.signal(signal.SIGINT, _sigint)

    logger.info(
        f"Collecting {max_frames} accepted frames | "
        f"blur_threshold={blur_threshold} | jpeg_quality={jpeg_quality}"
    )
    logger.info("─" * 60)

    try:
        while accepted < max_frames and not stop["now"]:

            # ── Capture ───────────────────────────────────────────────────
            frame = cam.capture_array("main")   # BGR despite RGB888 config
            total_captured += 1

            logger.info(
                f"Frame captured: {frame.shape[1]}x{frame.shape[0]} px, "
                f"dtype={frame.dtype}, range=[{frame.min():.1f}, {frame.max():.1f}]"
            )

            bgr = frame

            # ── Blur gate ─────────────────────────────────────────────────
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lv   = laplacian_variance(gray)

            if lv < blur_threshold:
                rejected_blur += 1
                logger.info(f"REJECTED blur: lv={lv:.1f} < threshold={blur_threshold:.1f}")
                continue

            logger.info(f"Blur OK: lv={lv:.1f}")

            # ── Face detection ────────────────────────────────────────────
            h, w = bgr.shape[:2]
            detector.setInputSize((w, h))
            start = time.time()
            _, faces = detector.detect(bgr)
            detection_time = time.time() - start

            if faces is None or len(faces) == 0:
                rejected_no_face += 1
                logger.info(f"No faces detected (took {detection_time:.2f}s)")
                continue

            # ── Pick largest face — mirrors face_service.py ───────────────
            coord_limit = 10 * max(h, w)
            face_locations = []
            for row in faces:
                arr = np.asarray(row, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    continue
                if float(np.max(np.abs(arr[:14]))) > coord_limit:
                    continue
                fx, fy, fw, fh = arr[:4]
                face_locations.append((int(fy), int(fx + fw), int(fy + fh), int(fx)))

            if not face_locations:
                rejected_no_face += 1
                logger.info(f"No valid face bbox (took {detection_time:.2f}s)")
                continue

            logger.info(f"Detected {len(face_locations)} face(s) (took {detection_time:.2f}s)")

            top, right, bottom, left = max(
                face_locations,
                key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]),
            )

            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)
            
            face_w = right - left
            face_h = bottom - top

            # ── Size gate ─────────────────────────────────────────────────
            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                rejected_too_small += 1
                logger.info(f"Face too small: {face_w}x{face_h} (minimum: {MIN_FACE_SIZE})")
                continue

            logger.info(f"Using face at ({left},{top})→({right},{bottom}), size: {face_w}x{face_h}")

            # ── ACCEPTED — encode and save both artefacts ─────────────────
            accepted += 1
            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}ms"

            full_bytes = encode_jpeg(bgr, jpeg_quality)
            with open(os.path.join(FULL_FRAMES_DIR, f"{ts}_full.jpg"), "wb") as f:
                f.write(full_bytes)
            full_sizes.append(len(full_bytes))

            face_crop  = bgr[top:bottom, left:right]
            face_bytes = encode_jpeg(face_crop, jpeg_quality)
            with open(os.path.join(FACE_CROPS_DIR, f"{ts}_face_{face_w}x{face_h}.jpg"), "wb") as f:
                f.write(face_bytes)
            face_sizes.append(len(face_bytes))

            logger.info(
                f"[{accepted:02d}/{max_frames}] SAVED | "
                f"full={len(full_bytes)//1024}KB | crop={len(face_bytes)//1024}KB | "
                f"lv={lv:.0f} | face={face_w}x{face_h}"
            )
            logger.info("─" * 60)

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        cam.stop()
        cam.close()
        logger.info("Picamera2 stopped.")

    if not full_sizes:
        logger.warning("No frames accepted. Try lowering --blur-threshold.")
        return

    # ── Summary ───────────────────────────────────────────────────────────
    def stats(sizes: list[int]) -> dict:
        arr = np.array(sizes)
        return {
            "min_kb":    round(int(arr.min())        / 1024, 1),
            "max_kb":    round(int(arr.max())        / 1024, 1),
            "mean_kb":   round(float(arr.mean())     / 1024, 1),
            "median_kb": round(float(np.median(arr)) / 1024, 1),
            "total_kb":  round(int(arr.sum())        / 1024, 1),
        }

    summary = {
        "config": {
            "frames_requested":  max_frames,
            "blur_threshold":    blur_threshold,
            "jpeg_quality":      jpeg_quality,
            "min_face_size_px":  MIN_FACE_SIZE,
        },
        "capture": {
            "total_raw_frames":   total_captured,
            "accepted":           accepted,
            "rejected_blur":      rejected_blur,
            "rejected_no_face":   rejected_no_face,
            "rejected_too_small": rejected_too_small,
        },
        "full_frame_bytes": stats(full_sizes),
        "face_crop_bytes":  stats(face_sizes),
    }

    with open(STATS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    fs = summary["full_frame_bytes"]
    cs = summary["face_crop_bytes"]

    print("\n" + "=" * 60)
    print("  NETWORK PAYLOAD BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Frames accepted        : {accepted}")
    print(f"  Rejected (blur)        : {rejected_blur}")
    print(f"  Rejected (no face)     : {rejected_no_face}")
    print(f"  Rejected (too small)   : {rejected_too_small}")
    print(f"  Total raw captured     : {total_captured}")
    print("-" * 60)
    print(f"  {'Metric':<28} {'Full frame':>12}  {'Face crop':>10}")
    print(f"  {'-'*28} {'-'*12}  {'-'*10}")
    for key, label in [("min_kb","Min"),("max_kb","Max"),("mean_kb","Mean"),("median_kb","Median")]:
        print(f"  {label:<28} {str(fs.get(key,'—'))+' KB':>12}  {str(cs.get(key,'—'))+' KB':>10}")
    print("-" * 60)
    ratio = round(fs["mean_kb"] / cs["mean_kb"], 1)
    print(f"  Full frame is ~{ratio}x larger than face crop on average")
    print(f"  At 1 frame/sec across 20 devices:")
    print(f"    Option 1 (full frame) : ~{round(fs['mean_kb']*20/1024, 1)} MB/s total LAN traffic")
    print(f"    Option 2a (face crop) : ~{round(cs['mean_kb']*20/1024, 2)} MB/s total LAN traffic")
    print("=" * 60)
    print(f"\n  Full stats : {STATS_FILE}")
    print(f"  Full frames: {FULL_FRAMES_DIR}/")
    print(f"  Face crops : {FACE_CROPS_DIR}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WPU network payload benchmark")
    parser.add_argument("--frames",         type=int,   default=30,    help="Accepted frames to collect")
    parser.add_argument("--blur-threshold", type=float, default=120.0, help="Laplacian variance threshold")
    parser.add_argument("--jpeg-quality",   type=int,   default=85,    help="JPEG quality 1-95")
    args = parser.parse_args()

    run_benchmark(
        max_frames     = args.frames,
        blur_threshold = args.blur_threshold,
        jpeg_quality   = args.jpeg_quality,
    )