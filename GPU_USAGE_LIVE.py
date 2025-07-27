import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
from collections import deque
import signal
import sys

gr3d_pattern = re.compile(r"GR3D_FREQ (\d+)%")
gpu_temp_pattern = re.compile(r"gpu@([\d.]+)C")
cpu_temp_pattern = re.compile(r"cpu@([\d.]+)C")
cpu_usage_pattern = re.compile(r"CPU \[(.*?)\]")

max_len = 100
gpu_usage = deque([0] * max_len, maxlen=max_len)
gpu_temp = deque([0] * max_len, maxlen=max_len)
cpu_usage = deque([0] * max_len, maxlen=max_len)
cpu_temp = deque([0] * max_len, maxlen=max_len)

tegrastats_proc = subprocess.Popen(
    ["tegrastats"],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True
)

def signal_handler(sig, frame):
    print("\nTerminating tegrastats...")
    tegrastats_proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def update_plot(frame):
    line_out = tegrastats_proc.stdout.readline()

    match = gr3d_pattern.search(line_out)
    gpu = int(match.group(1)) if match else 0
    gpu_usage.append(gpu)

    match = gpu_temp_pattern.search(line_out)
    g_temp = float(match.group(1)) if match else 0.0
    gpu_temp.append(g_temp)

    match = cpu_temp_pattern.search(line_out)
    c_temp = float(match.group(1)) if match else 0.0
    cpu_temp.append(c_temp)

    match = cpu_usage_pattern.search(line_out)
    if match:
        usage_str = match.group(1)
        core_usages = re.findall(r"(\d+)%@", usage_str)
        avg_cpu = sum(int(u) for u in core_usages) / len(core_usages) if core_usages else 0
    else:
        avg_cpu = 0
    cpu_usage.append(avg_cpu)

    gpu_line.set_ydata(gpu_usage)
    gpu_temp_line.set_ydata(gpu_temp)
    cpu_usage_line.set_ydata(cpu_usage)
    cpu_temp_line.set_ydata(cpu_temp)

    return gpu_line, gpu_temp_line, cpu_usage_line, cpu_temp_line

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

gpu_line, = ax1.plot(list(gpu_usage), label="GPU Load (%)")
gpu_temp_line, = ax1.plot(list(gpu_temp), label="GPU Temp (°C)")
ax1.set_ylim(0, 100)
ax1.set_title("Jetson GPU Usage & Temperature")
ax1.set_ylabel("GPU (%) / Temp (°C)")
ax1.grid(True)
ax1.legend()

cpu_usage_line, = ax2.plot(list(cpu_usage), label="CPU Load (%)")
cpu_temp_line, = ax2.plot(list(cpu_temp), label="CPU Temp (°C)")
ax2.set_ylim(0, 100)
ax2.set_title("Jetson CPU Usage & Temperature")
ax2.set_ylabel("CPU (%) / Temp (°C)")
ax2.set_xlabel("Time (samples)")
ax2.grid(True)
ax2.legend()

ani = animation.FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
plt.tight_layout()

try:
    plt.show()
finally:
    tegrastats_proc.terminate()