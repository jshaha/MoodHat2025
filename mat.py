import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time  # Track real-world time
from BCI import *
import matplotlib.pyplot as plt
import random


source = BCI()
source.launch_server_osc()
newPipe = Pipe(1, len(source.BCI_params["channel_names"]), source.store)
newPipe.launch_server()

# Create the figure and two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Settings
num_lines = len(source.BCI_params["channel_names"])  # Number of lines per graph
time_window = 10  # Show last 10 seconds of data
colors = ["red", "blue", "green", "purple"]  # Different colors for each line

# Initialize lists for real-time x and y data
x_data = [time.time()]  # Start with current time
y_data1 = [[] for i in range(num_lines)]  # Sine wave values (one per line)
y_data2 = [[] for i in range(num_lines)]  # Cosine wave values (one per line)

# Create multiple line objects for both graphs
lines1 = [ax1.plot(x_data, y_data1[i], color=colors[i], label= source.BCI_params["channel_names"][i])[0]for i in range(num_lines)]
lines2 = [ax2.plot(x_data, y_data2[i], color=colors[i], label=source.BCI_params["channel_names"][i])[0] for i in range(num_lines)]

# Customize the plots
for ax in (ax1, ax2):
    ax.set_ylim(-1250, 1250)  # Fixed y-axis range
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hertz")
    ax.legend()
    ax.grid(False)

ax1.set_title("Input data")
ax2.set_title("Output data")

# Update function for animation
def update(frame):
    current_time = time.time()  # Get the current time

    # Generate new values for all 4 lines
    new_y1 = [newPipe.store[0][i].get() for i in range(num_lines)]
    new_y2 = [newPipe.store[0][i].get() for i in range(num_lines)]

    # Append new values to data lists
    x_data.append(current_time)
    for i in range(num_lines):
        y_data1[i].append(new_y1[i])
        y_data2[i].append(new_y2[i])

    # Keep only the last 'time_window' seconds of data
    while x_data[-1] - x_data[0] > time_window:
        x_data.pop(0)
        for i in range(num_lines):
            y_data1[i].pop(0)
            y_data2[i].pop(0)

    # Update all four lines for both graphs
    for i in range(num_lines):
        lines1[i].set_data(x_data, y_data1[i])
        lines2[i].set_data(x_data, y_data2[i])

    # Dynamically adjust x-axis range
    ax1.set_xlim(x_data[0], x_data[-1])
    ax2.set_xlim(x_data[0], x_data[-1])

    return lines1 + lines2  # Return all line objects

# Run animation
ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
plt.show()
