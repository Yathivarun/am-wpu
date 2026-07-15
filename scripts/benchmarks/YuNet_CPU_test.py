import cv2
import time
import psutil
import os

from wpu_client.paths import MODELS_DIR

# Config
MODEL_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
IMAGE_PATH = "passport_photo.jpeg"  # drop a local test image next to this script
INPUT_SIZE = (320, 320)
ITERATIONS = 100

# Load Model
yunet = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    INPUT_SIZE,
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# Load Image
frame = cv2.imread(IMAGE_PATH)
frame = cv2.resize(frame, INPUT_SIZE)

# Process Handle
process = psutil.Process(os.getpid())

# Warmup
for _ in range(10):
    yunet.detect(frame)

# Init CPU tracking
process.cpu_percent(interval=None)  
start_cpu = process.cpu_times() 

# Benchmark
latencies = []
start_total = time.time()

for i in range(ITERATIONS):
    start = time.time()
    _, faces = yunet.detect(frame)
    end = time.time()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

end_total = time.time()
end_cpu = process.cpu_times()

# Metrics
avg_latency = sum(latencies) / len(latencies)
min_latency = min(latencies)
max_latency = max(latencies)

total_time = end_total - start_total
fps = ITERATIONS / total_time

# CPU Usage
cpu_time_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
wall_time = total_time

cpu_usage = (cpu_time_used / wall_time) * 100

# Output
print("\n===== YuNet Benchmark Results =====")
print(f"Input Size: {INPUT_SIZE}")
print(f"Iterations: {ITERATIONS}")
print(f"-----------------------------------")
print(f"Avg Latency: {avg_latency:.2f} ms")
print(f"Min Latency: {min_latency:.2f} ms")
print(f"Max Latency: {max_latency:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"CPU Usage: {cpu_usage:.2f} %")
print("===================================\n")