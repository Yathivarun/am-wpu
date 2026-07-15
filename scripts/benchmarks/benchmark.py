import time
import numpy as np
import onnxruntime as ort
import psutil
import os
import threading

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilenetv2_mcp.onnx")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)

# Warmup
for _ in range(3):
    session.run(None, {input_name: dummy_input})

# High-frequency CPU sampler running in background thread
cpu_log = []
stop_flag = threading.Event()

def cpu_sampler():
    while not stop_flag.is_set():
        # cpu_log.append((time.perf_counter(), psutil.cpu_percent(percpu=True)))
        cpu_log.append((time.perf_counter(), sum(psutil.cpu_percent(percpu=True))))
        time.sleep(0.05)  # sample every 50ms

sampler = threading.Thread(target=cpu_sampler)
sampler.start()

# Benchmark
latencies = []
for i in range(30):
    t0 = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    t1 = time.perf_counter()
    latencies.append((t0, t1))

stop_flag.set()
sampler.join()

print(f"Latency — mean: {np.mean([e-s for s,e in latencies])*1000:.1f}ms")
print(f"Peak CPU sampled: {max(c for _,c in cpu_log):.1f}%")
print(f"Mean CPU sampled: {np.mean([c for _,c in cpu_log]):.1f}%")