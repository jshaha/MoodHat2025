import queue
import threading
import time
from BCI import MovingAverageFilter

def test_moving_average_filter():
    # Create a single input queue (the filter expects one queue containing data for all channels)
    input_queue = queue.Queue()
    
    # Number of channels to simulate
    num_channels = 4
    
    def feeder():
        """Feed sample data into the input queue"""
        value = 1
        while True:
            # Create a list with values for all channels
            # For testing, we'll use the same value for all channels
            channel_data = [value] * num_channels
            input_queue.put(channel_data)
            value += 1
            time.sleep(0.01)
    
    # Start the feeder thread
    t = threading.Thread(target=feeder)
    t.daemon = True
    t.start()
    
    # Create the filter with correct parameters
    avg_filter = MovingAverageFilter(
        num_input_channels=num_channels,
        input_store=input_queue,
        window_size=8
    )
    
    # Start the filter in a separate thread since it has an infinite loop
    filter_thread = threading.Thread(target=avg_filter.action)
    filter_thread.daemon = True
    filter_thread.start()
    
    # Wait for enough data to accumulate
    time.sleep(0.5)  # Wait for at least window_size samples
    
    # Get and print results from output queues
    for i in range(5):  # Print 5 rows of output
        # Get one value from each output queue
        row = [q.get() for q in avg_filter.output_queues]
        print(f"Output row {i+1}: {row}")
        time.sleep(0.1)  # Short delay between reads

if __name__ == "__main__":
    test_moving_average_filter()