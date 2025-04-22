from BCI import csvOutput  # Assuming it's in its own file

import queue
import threading
import time
import numpy as np

# Simulate 4-channel EEG
queues = [queue.Queue() for _ in range(4)]

# Fill with random data
def simulate_data():
    while True:
        for q in queues:
            q.put(np.random.randn())
        time.sleep(0.01)

threading.Thread(target=simulate_data, daemon=True).start()

# Start CSV block
csv_block = csvOutput(num_input_channels=4, input_store=queues, file_name="test_output.csv")
csv_block.action()

time.sleep(5)
print("Check test_output.csv!")