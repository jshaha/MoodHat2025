import queue
import threading
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pythonosc import dispatcher, osc_server
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path


# Determine the directory containing the module
relative_path = Path("../../BCI.py")
module_directory = relative_path.parent.resolve()

# Add the directory to sys.path
sys.path.insert(0, str(module_directory))

# Now you can import the module
from BCI import *





firsttime = None # record the first timestamp during recording


headset = BCI("MuseS") # ["TP9", "AF7", "AF8", "TP10"]
headset_server, headset_server_thread = headset.launch_server()

pipe_obj = Pipe(1, headset.no_of_channels, headset.store, headset.time_store)
pipe_obj.launch_server()



# # Start a thread to calculate the averages
# average_thread = threading.Thread(target=calculate_averages)
# average_thread.daemon = True
# average_thread.start()

# Real-time plotting setup

def store_empty(store_list):
    return any([item.empty() for item in store_list])

newstore = [queue.Queue() for _ in range(headset.no_of_channels)]

for j in range(headset.no_of_channels):
                    value = headset.store[j].get()
                    newstore[j].put(value)

final_output_store = [newstore, pipe_obj.store[1]]
number_of_plots = pipe_obj.no_of_outputs
number_of_lines = pipe_obj.no_of_input_channels
list_of_labels = headset.channel_names

# fig, axes_list = plt.subplots(number_of_plots, 1)
axes_list = [None] * number_of_plots
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
axes_list = [ax1, ax2]

ax1.set_title("Raw EEG Data")
ax2.set_title("Processed Pipe Data")

xdata = [[] for _ in range(number_of_plots)]
ydata_matrix = [[[] for _ in range(number_of_lines)] for _ in range(number_of_plots)]
plot_matrix = [[None for _ in range(number_of_lines)] for _ in range(number_of_plots)]


for i in range(number_of_plots):
    for j in range(number_of_lines):
        a, = axes_list[i].plot([],[], label=list_of_labels[j])
        plot_matrix[i][j] = a
    axes_list[i].legend()

# print("HI THIS IS A: ", type(plot_matrix[0][0]))

def init():
    for i in range(number_of_plots):
        axes_list[i].set_xlim(0, 40)
        axes_list[i].set_ylim(-1200, 1200)
        for j in range(number_of_lines):
            plot_matrix[i][j].set_data([], [])
    return [line for sublist in plot_matrix for line in sublist]
    # return ln_tp9, ln_af7, ln_af8, ln_tp10 # does init() require a return when blit=True is not set?
    

def update(frame):
    global firsttime
    
    # Check if both data sources have data available
    raw_has_data = not store_empty(headset.store)
    processed_has_data = not store_empty(final_output_store[1])
    
    # Only update when both have data (to keep them synchronized)
    if raw_has_data and processed_has_data:
        # Update frame counter for both plots
        current_frame = xdata[0][-1] + 1 if xdata[0] else frame
        xdata[0].append(current_frame)
        xdata[1].append(current_frame)
        
        # Get raw data for first plot
        for j in range(number_of_lines):
            if not headset.store[j].empty():
                raw_value = headset.store[j].get()
                ydata_matrix[0][j].append(raw_value)
        
        # Get processed data for second plot
        for j in range(number_of_lines):
            if not final_output_store[1][j].empty():
                processed_value = final_output_store[1][j].get()
                ydata_matrix[1][j].append(processed_value)
        
        # Maintain window size for both plots
        if len(xdata[0]) > 40:
            xdata[0].pop(0)
            xdata[1].pop(0)
            for j in range(number_of_lines):
                if len(ydata_matrix[0][j]) > 0:
                    ydata_matrix[0][j].pop(0)
                if len(ydata_matrix[1][j]) > 0:
                    ydata_matrix[1][j].pop(0)
        
        # Update axis limits
        if xdata[0]:
            axes_list[0].set_xlim(xdata[0][0], xdata[0][-1])
            axes_list[1].set_xlim(xdata[1][0], xdata[1][-1])
        
        # Update plot data
        for j in range(number_of_lines):
            plot_matrix[0][j].set_data(xdata[0], ydata_matrix[0][j])
            plot_matrix[1][j].set_data(xdata[1], ydata_matrix[1][j])
    
    # Return all plots to be updated
    return [line for sublist in plot_matrix for line in sublist]
        


# Create a ThreadPoolExecutor to handle averaging in a separate thread
# with ThreadPoolExecutor(max_workers=2) as executor:
#     # Submit the averaging task to the executor
#     executor.submit(calculate_averages)
    
ani = animation.FuncAnimation(fig, update, frames=range(1000), init_func=init, interval=1)
plt.show()

try:
    while True:
        pass  # Keep the main thread running
except KeyboardInterrupt:
    print("Exiting...")
    headset_server.shutdown()
    headset_server_thread.join()
    plt.close()