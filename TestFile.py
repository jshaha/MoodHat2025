from BCI import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import queue
import threading
import time
import numpy as np

NUM_CHANNELS = 4

# === Set up real Muse source and Pipe ===
source = BCI("MuseS")
server, serverThread = source.launch_server_osc()
newPipe = Pipe(1, len(source.BCI_params["channel_names"]), source.store)
newPipe.launch_server()

# === Set up simulated/fake data fallback ===
fake_store = [[queue.Queue() for _ in range(NUM_CHANNELS)]]

# Fill fake_store with random values continuously
def simulate_fake_data():
    while True:
        for i in range(NUM_CHANNELS):
            fake_store[0][i].put(random.uniform(-1, 1))
        time.sleep(0.05)  # ~20Hz, similar to Muse sample rate

fake_thread = threading.Thread(target=simulate_fake_data, daemon=True)
fake_thread.start()

no_of_channels = 4
fig, ax = plt.subplots()
x_data = list(range(100))
y_data = [[0] * 100 for _ in range(no_of_channels)]
lines = [ax.plot(x_data, y_data[ch], label=f'Channel {ch+1}')[0] for ch in range(no_of_channels)]

# Set axis limits
ax.set_xlim(0, 100)  # Adjust according to your needs
ax.set_ylim(-100, 100)   # Adjust range based on data
ax.set_xlabel("Time")
ax.set_ylabel("Brain Waves")
ax.set_title("Real-Time BCI Data Stream")
ax.legend()

def update(frame):
    # Check if real data is available for all channels
    use_real = all(not newPipe.store[0][i].empty() for i in range(NUM_CHANNELS))

    if use_real:
        data_source = newPipe.store
    else:
        data_source = fake_store

    for ch in range(NUM_CHANNELS):
        try:
            if not data_source[0][ch].empty():
                value = data_source[0][ch].get_nowait()
            else:
                value = y_data[ch][-1]  # Repeat last value if none available
        except queue.Empty:
            value = y_data[ch][-1]

        # Update y_data and plot
        y_data[ch].append(value)
        y_data[ch].pop(0)
        lines[ch].set_ydata(y_data[ch])

    return lines

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 1000), interval=100, blit=False, cache_frame_data=False)
plt.show()