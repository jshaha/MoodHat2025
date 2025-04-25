import time
import numpy as np
from abc import ABC, abstractmethod
import queue
import threading
from utils import generate_random_string, nanVal
from scipy.integrate import simps
import scipy.signal as signal
import argparse
from pythonosc import dispatcher as disp, osc_server
import os
import csv
# import pylsl
import mne


class BCI:
    '''
        Class to manage collection of the BCI data and store into a buffer store. Used as the input of
        the pipeline To receive data from the headset and store it into a list of storage queues for which
        later blocks of the pipeline can pull from.. 
        NOTE: This version is set up for Muse S, using Petal Metrics' tool for handling the streaming protocol,
        and also supports artificial sine wave generation for testing.
        The current version works with OSC, LSL streaming is a TODO.

        Class Variable:
            - sampling_rate: sampling rate of the device
            - streaming_software: software used for heandling the streaming (Petals for Muse S) (TODO: make this an option arguments)
            - streaming_protocol: streaming protocol used by headset
            - channel_names: list of names of channels, used for labels
            - no_of_channels: number of channels for the device. Channels are defined as an independent stream of data that most processing blocks will treat as independent streams to compute the same operations on in parallel
            - store: a list of queues where input data from the datastreams are added to, which later blocks can get() from
            - time_store: single queue that holds timestamp data # TODO: make this function
            - launch_server: function to initialize and start streaming data into self.store, determined by self.streaming_protocol
            - is_artificial: boolean flag to indicate if using artificial signals instead of real BCI data
            - artificial_signal_params: parameters for the artificial signal generation
    '''
    def __init__(self, BCI_name="MuseS", BCI_params={}, is_artificial=False, artificial_signal_params=None):
        self.name = BCI_name
        self.is_artificial = is_artificial
        
        if self.name == "MuseS":
            self.BCI_params = {"sampling_rate": 256, "channel_names":["TP9", "AF7", "AF8", "TP10"], 
                               "streaming_software":"Petals", "streaming_protocol":"OSC", "cache_size":256*30}
            if BCI_params:
                for i, j in BCI_params.items():  # Fixed the iteration syntax here
                    self.BCI_params[i] = j
        else:
            raise Exception("Unsupported BCI board") # change this when adding other headsets
        
        self.sampling_rate = self.BCI_params["sampling_rate"]
        self.streaming_software = self.BCI_params["streaming_software"] # # TODO: add as optional argument
        self.streaming_protocol = self.BCI_params["streaming_protocol"] # TODO: mandatory argument, add error handling
        self.channel_names = self.BCI_params["channel_names"] # TODO: add error handling -> warning for non standard channel names
        self.no_of_channels = len(self.channel_names) # TODO: add error handling
        self.store = [queue.Queue() for i in range(self.no_of_channels)]
        self.time_store = queue.Queue()
        
        # Default artificial signal parameters if none provided
        if artificial_signal_params is None and is_artificial:
            self.artificial_signal_params = {
                "frequencies": [10.0, 12.0, 15.0, 8.0],  # Hz - different freq for each channel
                "amplitudes": [10.0, 15.0, 10.0, 12.0],  # µV - different amplitude for each channel
                "noise_level": 2.0,  # µV - background noise level
                "dc_offset": [0.0, 0.0, 0.0, 0.0]  # DC offset for each channel
            }
        else:
            self.artificial_signal_params = artificial_signal_params
            
        if self.streaming_protocol == 'LSL' and not is_artificial:
            self.launch_server = self.launch_server_lsl
        elif self.streaming_protocol == 'OSC' and not is_artificial:
            self.launch_server = self.launch_server_osc
        elif is_artificial:
            self.launch_server = self.launch_artificial_signal_generator
        
        # Time tracking for artificial signal generation
        self.start_time = 0
        self.current_sample = 0
        self.artificial_thread = None
        self.stop_flag = False

    def action(self):
        pass # not required for BCI object

    def handle_osc_message(self, address, *args):
        '''
        Receives messages through the OSC channel and adds them to a list of queues.
        ''' 
        self.store[0].put(args[5])
        self.store[1].put(args[6])
        self.store[2].put(args[7])
        self.store[3].put(args[8])
        self.time_store.put(args[3] + args[4])

    def launch_server_osc(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--ip', type=str, required=False,
                            default="127.0.0.1", help="The ip to listen on")
        parser.add_argument('-p', '--udp_port', type=str, required=False, default=14739,
                            help="The UDP port to listen on")
        parser.add_argument('-t', '--topic', type=str, required=False,
                            default='/PetalStream/eeg', help="The topic to print")
        args = parser.parse_args()

        dispatcher = disp.Dispatcher()
        dispatcher.map(args.topic, self.handle_osc_message)

        server = osc_server.ThreadingOSCUDPServer((args.ip, args.udp_port), dispatcher)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        return server, server_thread
        
    def generate_artificial_samples(self):
        """
        Generate artificial sine wave samples with optional noise and DC offset
        """
        t = self.current_sample / self.sampling_rate
        samples = []
        
        for i in range(self.no_of_channels):
            # Generate sine wave for this channel with the specified frequency and amplitude
            freq = self.artificial_signal_params["frequencies"][i]
            amp = self.artificial_signal_params["amplitudes"][i]
            dc = self.artificial_signal_params["dc_offset"][i]
            
            # Calculate the sine wave value at this time point
            sine_val = amp * np.sin(2 * np.pi * freq * t)
            
            # Add random noise
            if "noise_level" in self.artificial_signal_params:
                noise = np.random.normal(0, self.artificial_signal_params["noise_level"])
                sine_val += noise
                
            # Add DC offset
            sine_val += dc
            
            samples.append(sine_val)
            print("samples:", samples)
        return samples
    
    def artificial_signal_loop(self):
        """
        Main loop for generating artificial signals at the appropriate sampling rate
        """
        self.start_time = time.time()
        self.current_sample = 0
        
        while not self.stop_flag:
            # Calculate the expected time for this sample
            expected_time = self.start_time + (self.current_sample / self.sampling_rate)
            
            # Generate the sample
            samples = self.generate_artificial_samples()
            
            # Add samples to the queues
            for i, sample in enumerate(samples):
                self.store[i].put(sample)
            
            # Add timestamp to time_store
            current_time = time.time()
            self.time_store.put(current_time)
            
            # Increment the sample counter
            self.current_sample += 1
            
            # Sleep until it's time for the next sample
            next_sample_time = self.start_time + ((self.current_sample) / self.sampling_rate)
            sleep_time = max(0, next_sample_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def launch_artificial_signal_generator(self):
        """
        Start the artificial signal generator thread
        """
        self.stop_flag = False
        self.artificial_thread = threading.Thread(target=self.artificial_signal_loop)
        self.artificial_thread.daemon = True
        self.artificial_thread.start()
        
        return None, self.artificial_thread
    
    def stop_artificial_signal_generator(self):
        """
        Stop the artificial signal generator
        """
        if self.artificial_thread:
            self.stop_flag = True
            self.artificial_thread.join(timeout=1.0)
            self.artificial_thread = None
	# def launch_server_lsl(self):
	# 	'''
	# 	Receives messages through the OSC channel and adds them to a list of queues.
	# 	TODO: broken, requires fixing.
	# 	''' 
	# 	# TODO: take code from Petal metrics and implement streaming with LSL
	# 	parser = argparse.ArgumentParser()
	# 	parser.add_argument('-n', '--stream_name', type=str, required=True,
	# 						default='PetalStream_eeg', help='the name of the LSL stream')
	# 	args = parser.parse_args()
	# 	print(f'looking for a stream with name {args.stream_name}...')
	# 	streams = pylsl.resolve_stream('name', args.stream_name)
	# 	if len(streams) == 0:
	# 		raise RuntimeError(f'Found no LSL streams with name {args.stream_name}')
	# 	inlet = pylsl.StreamInlet(streams[0])

	# 	...


class Pipe:
    '''
    Connects one or more ProcessingBlocks together. Creates a pipeline of reordering
    and moving average filtering.
    '''
    def __init__(self, no_of_outputs, no_of_input_channels, input_store, time_store) -> None:
        self.no_of_outputs = no_of_outputs
        self.no_of_input_channels = no_of_input_channels
        self.input_store = input_store
        self.time_store = time_store
        self.name = "PIPE_" + generate_random_string()
        
        # Create intermediate queues for the reordered data
        self.reordered_store = [queue.Queue() for _ in range(no_of_input_channels)]
        self.reordered_time_store = queue.Queue()
        
        # Create output queues for filtered data
        self.store = [[queue.Queue() for j in range(no_of_input_channels)] for i in range(no_of_outputs)]
        
        # Create the processing blocks
        self.reorder_block = ReorderingBlock(
            no_of_input_channels=self.no_of_input_channels, 
            input_store=self.input_store,
            time_store=self.time_store,
            buffer_size=15
        )
        
        self.moving_avg = MovingAverageFilter(
            no_of_input_channels=self.no_of_input_channels, 
            input_store=self.reorder_block.store,  # Connect to reorder block's output
            window_size=8
        )
        self.notch_filter = NotchFilter(
              no_of_input_channels=self.no_of_input_channels,
                input_store=self.moving_avg.store,
				notch_frequency=60,
				sampling_frequency=500,
				quality_factor=30 )
        
        self.psd_processor = PSDProcessor(
            no_of_input_channels=self.no_of_input_channels,
            sampling_frequency=500,  # Match your actual sampling rate
            window_size=50,        # Adjust as needed for frequency resolution
            input_store=self.notch_filter.store,  # Process after notch filtering
            overlap=0.75             # 75% overlap for smoother updates
        )
        
    def action(self):
        '''
        Forward data from the moving average filter to all output queues
        '''
        while True:
            try:
                # For each channel, forward data from moving average to all outputs
                for j in range(self.no_of_input_channels):
                    print()
                    value = self.notch_filter.store[j].get()
                    
                    for i in range(self.no_of_outputs):
                        self.store[i][j].put(value)
            except KeyboardInterrupt:
                print(f"Closing {self.name} thread...")
                break

    def launch_server(self):
        # Start the reordering block
        self.reorder_block.launch_server()
        print(f"{self.reorder_block.name} thread started...")
        
        # Start the moving average filter
        self.moving_avg.launch_server()
        print(f"{self.moving_avg.name} thread started...") 
        
        # Start notch filter
        self.notch_filter.launch_server()
        print(f"{self.notch_filter.name} thread started...")
        
        self.psd_processor.launch_server()
        print(f"{self.psd_processor.name} thread started...")

        # Start the pipe's own thread to forward output
        pipe_thread = threading.Thread(target=self.action)
        pipe_thread.daemon = True
        pipe_thread.start()
        print(f"{self.name} thread started...")
        

	# TODO: add functionality to stop the pipe if any output is not ready for loading yet

class ProcessingBlock(ABC):
	'''
	Abstract class for processing block. Should have an _init_ function, an action function, a process function to start the processing thread
	'''
	# def __init__(self, cache_dim=(256*5)):
		# self.input_queue = input_queue # reference to queue to pool data from into own cache
		# self.output_queue = output_queue 

	@abstractmethod
	def __init__():
		pass

	def launch_server(self):
		average_thread = threading.Thread(target=self.action)
		average_thread.daemon = True
		average_thread.start()
		print(f"{self.name} thread started...")
	
	def action(self):
		pass



class ReorderingBlock(ProcessingBlock):
    """
    Reorders incoming EEG samples based on their timestamps.
    Adapted to work with the Pipe system where timestamps and channel data
    come from separate queues.
    """
    
    def __init__(self, no_of_input_channels, input_store, time_store, buffer_size=15):
        self.no_of_input_channels = no_of_input_channels
        self.input_store = input_store  # List of queues, one per channel
        self.time_store = time_store    # Queue of timestamps
        self.buffer_size = buffer_size
        self.name = "SampleReorder_" + generate_random_string()
        
        # Create output queues for reordered data
        self.store = [queue.Queue() for _ in range(self.no_of_input_channels)]
        self.reordered_time_store = queue.Queue()
        
        # Buffer for collecting samples before reordering
        self.buffer = []
        
        # For monitoring
        self.samples_processed = 0
        
    def action(self):
        """Process and reorder samples based on timestamps."""
        while True:
            try:
                # Collect buffer_size samples
                while len(self.buffer) < self.buffer_size:
                    # Get timestamp
                    timestamp = self.time_store.get(timeout=1.0)
                    
                    # Get one sample from each channel
                    channel_data = []
                    for i in range(self.no_of_input_channels):
                        channel_data.append(self.input_store[i].get(timeout=1.0))
                    
                    # Store sample with timestamp as sorting key
                    self.buffer.append((timestamp, channel_data))
                
                # Sort the buffer by timestamp
                self.buffer.sort(key=lambda x: x[0])
                
                # Output reordered samples
                for timestamp, channel_data in self.buffer:
                    # Put timestamp in time store
                    self.reordered_time_store.put(timestamp)
                    
                    # Put channel data in respective output queues
                    for i in range(self.no_of_input_channels):
                        self.store[i].put(channel_data[i])
                    
                    self.samples_processed += 1
                
                # Clear the buffer for the next batch
                self.buffer = []
                
                # Periodically log stats
                if self.samples_processed % 1000 == 0:
                    print(f"{self.name}: Processed {self.samples_processed} samples")
                    
            except queue.Empty:
                # No data available, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                time.sleep(0.1)
                
class MovingAverageFilter(ProcessingBlock):
    """
    Calculates moving average for each channel independently.
    Adapted to work with the Pipe system.
    """
    
    def __init__(self, no_of_input_channels, input_store, window_size=8):
        self.no_of_input_channels = no_of_input_channels
        self.input_store = input_store  # List of queues, one per channel
        self.window_size = window_size
        self.name = "MovingAvg_" + generate_random_string()
        
        # Create output queues
        self.store = [queue.Queue() for _ in range(self.no_of_input_channels)]
        
        # Initialize buffers for each channel
        self.buffers = [[] for _ in range(self.no_of_input_channels)]
        
    def action(self):
        """Process incoming data with a moving average filter."""
        while True:
            try:
                # Process each channel independently
                for channel_idx in range(self.no_of_input_channels):
                    # Get new data point
                    if not self.input_store[channel_idx].empty():
                        value = self.input_store[channel_idx].get()
                        
                        # Add to buffer
                        buffer = self.buffers[channel_idx]
                        buffer.append(value)
                        
                        # If buffer is full, calculate average and output
                        if len(buffer) >= self.window_size:
                            # Calculate average
                            average = sum(buffer) / self.window_size
                            
                            # Put the result in the output queue
                            self.store[channel_idx].put(average)
                            
                            # Remove oldest value
                            buffer.pop(0)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                time.sleep(0.1)


class ArrayIntegrator(ProcessingBlock):
	'''
	Integrates the array for each channel and outputs a single value per channel. Does not integrate over time

	Class Variables:
	- no_of_input_channels: number of input channels
	- x_axis_resolution: dx of the x-axis values, assumed uniform spacing
	- no_of_inputs: number of input values (without regarding channel count)
	- prior_valid: points to the validity of the prior block (need to be precomputed for validity of multiple inputs) (TODO: not implemented)
	- array_of_inputs <-> *args : list of stores to input from to integrate
	- name: name of block
	- cache: temporary store (no_of_input_channels x no_of_inputs)
	- store: output store (1 x no_of_input_channels)
	- valid: validity of output (TODO: not implemented)
	'''
	def __init__(self, no_of_input_channels, x_axis_resolution, no_of_inputs, prior_valid, *args):
		...



class Ratio(ProcessingBlock):
	'''
	Calculates the ratio between two values over multiple channels. Outputs input1 / input2

	Class variables:
	- no_of_input_channels: number of channels in inputs (must be equal)
	- input1: source for the numerator input
	- input2: source for the denominator input
	- store: queue for the output stream
	- valid: indicator for valid input
	'''
	def __init__(self, no_of_input_channels, input_store1, input_store2, prior_valid1, prior_valid2):
		self.no_of_input_channels = no_of_input_channels
		self.input_store1 = input_store1
		self.input_store2 = input_store2
		self.prior_valid1 = prior_valid1
		self.prior_valid2 = prior_valid2

	def ratio(self):
		if (self.prior_valid1 and self.prior_valid2):
			return self.input_store1 / self.input_store2



class NotchFilter(ProcessingBlock):
    '''
    Applies a Notch Filter on the data stream. Apply this on the input data 
    (easiest to do this before applying any other processing blocks)
    '''
    def __init__(self, no_of_input_channels, input_store, notch_frequency, sampling_frequency, quality_factor):
        self.no_of_input_channels = no_of_input_channels
        self.input_store = input_store
        self.notch_frequency = notch_frequency
        self.sampling_frequency = sampling_frequency
        self.quality_factor = quality_factor
        
        self.name = "NotchFilter_" + generate_random_string()
        
        # Output queue for each channel
        self.store = [queue.Queue() for _ in range(self.no_of_input_channels)]
        
        # Buffer to accumulate data for filtering
        self.buffer_size = 128  # Adjust based on your needs
        self.buffers = [[] for _ in range(self.no_of_input_channels)]
    
    def action(self):
        while True:
            try:
                # Process each channel
                for j in range(self.no_of_input_channels): 
                    # Get new data if available
                    if not self.input_store[j].empty():
                        value = self.input_store[j].get()
                        
                        # Add to buffer
                        self.buffers[j].append(value)
                         # If buffer is full, apply notch filter
                        if len(self.buffers[j]) >= self.buffer_size:
                            # Convert to numpy array for filtering
                            data_array = np.array(self.buffers[j])
                            
                            # Apply notch filter
                            filtered_data = mne.filter.notch_filter(
                                data_array, 
                                self.sampling_frequency, 
                                self.notch_frequency,
                                filter_length='auto', 
                                notch_widths=None, 
                                trans_bandwidth=1, 
                                method='fir',
                                phase='zero', 
                                fir_window='hamming'
                            )
                            # Output the filtered value and remove oldest value
                            self.store[j].put(filtered_data[-1])
                            self.buffers[j].pop(0)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.001)
                
            except KeyboardInterrupt:
                print(f"Closing {self.name} thread...")
                break
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                time.sleep(0.1)

class csvOutput:
    '''
    Takes incoming data and saves it into a CSV format.
    '''
    def __init__(self, num_input_channels, input_store, file_name="eeg_data.csv"):
        self.num_input_channels = num_input_channels
        self.input_store = input_store  
        self.file_name = file_name
        self.name = "CSV_Output"

        # Remove old file if it exists
        if os.path.exists(file_name):
            os.remove(file_name)

        # Create header
        with open(self.file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f"Channel_{i}" for i in range(self.num_input_channels)]
            writer.writerow(header)


    def action(self):
        while True:
            try:
                # Get one sample from each channel
                row = [self.input_store[i].get() for i in range(self.num_input_channels)]

                # Append to CSV
                with open(self.file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            except Exception as e:
                print(f"[csvOutput] Error writing to CSV: {e}")



class PSDProcessor(ProcessingBlock):
    """
    Calculates Power Spectral Density and band powers for each channel.
    Works with continuous data flows from queues using SciPy's Welch method.
    """
    
    def __init__(self, no_of_input_channels, sampling_frequency, window_size, input_store, overlap=0.5):
        self.no_of_input_channels = no_of_input_channels
        self.sampling_frequency = sampling_frequency
        self.window_size = window_size
        self.overlap = overlap
        self.input_store = input_store
        self.name = "PSD_" + generate_random_string()
        
        # Calculate step size based on overlap
        self.step_size = int(self.window_size * (1 - self.overlap))
        if self.step_size < 1:
            self.step_size = 1
	
        # Define frequency bands of interest for emotion detection
        self.bands = {
            'delta': (1, 4),    # Deep sleep, unconsciousness
            'theta': (4, 8),    # Drowsiness, meditation
            'alpha': (8, 13),   # Relaxation, calmness
            'beta': (13, 30),   # Active thinking, focus
            'gamma': (30, 45)   # Higher processing
        }
        
        # Create output queues - will contain tuples of (frequencies, psd_values)
        self.psd_store = [queue.Queue() for _ in range(self.no_of_input_channels)]
        
        # Create output queue for band powers - will contain dictionaries
        self.band_power_store = queue.Queue()
        
        # Initialize buffers for each channel
        self.buffers = [[] for _ in range(self.no_of_input_channels)]
        
    def action(self):
        """Process incoming data with Welch's method for PSD estimation using SciPy."""
        print(f"{self.name} thread started...")
        while True:
            try:
                new_data_processed = False

                # Process each channel independently
                for channel_idx in range(self.no_of_input_channels):
                    # Get new data point
                    if not self.input_store[channel_idx].empty():
                        value = self.input_store[channel_idx].get()
                        
                        # Add to buffer
                        buffer = self.buffers[channel_idx]
                        buffer.append(value)
                        
                        # If buffer has enough samples, calculate PSD
                        if len(buffer) >= self.window_size:
                            # Get window of data
                            data_window = np.array(buffer)
                            
                            # Use scipy.signal.welch to calculate PSD
                            freqs, psd_values = signal.welch(
                                data_window, 
                                fs=self.sampling_frequency,
                                window='hann',
                                nperseg=self.window_size,
                                noverlap=int(self.window_size * self.overlap),
                                scaling='density'
                            )
                            print(f"Frequency range: {min(freqs)}-{max(freqs)} Hz, Max power: {max(psd_values)}")
                            
                            # Put the raw PSD result in its output queue
                            self.psd_store[channel_idx].put((freqs, psd_values))
                            
                            # Remove oldest value to maintain sliding window
                            buffer.pop(0)
                            new_data_processed = True
                
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
           
		        
class bandPower(ProcessingBlock):
    def __init__(self, no_of_channels, psd_store):
        self.no_of_channels = no_of_channels
        self.psd_store = psd_store  # List of queues containing (freqs, psd_values) tuples
        self.name = "BandPower_" + generate_random_string()
        
        # Define frequency bands of interest for emotion detection
        self.bands = {
            'delta': (1, 4),    # Deep sleep, unconsciousness
            'theta': (4, 8),    # Drowsiness, meditation
            'alpha': (8, 13),   # Relaxation, calmness
            'beta': (13, 30),   # Active thinking, focus
            'gamma': (30, 45)   # Higher processing
        }
        
        # Create output queue for band powers - will contain dictionaries
        self.band_power_store = queue.Queue()
        
    def action(self):
        """Process PSD data to calculate band powers."""
        print(f"{self.name} thread started...")
        while True:
            try:
                # Dictionary to hold band powers for all channels
                all_band_powers = {}
                new_data_processed = False

                # Process each channel independently
                for channel_idx in range(self.no_of_channels):
                    # Check if there's new PSD data
                    if not self.psd_store[channel_idx].empty():
                        # Get PSD data
                        freqs, psd_values = self.psd_store[channel_idx].get()
                        
                        # Calculate band powers
                        for band_name, (fmin, fmax) in self.bands.items():
                            # Find indices corresponding to frequency range
                            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                            
                            # Calculate band power (area under the curve)
                            if np.any(idx_band):  # Only calculate if we have data in this range
                                power = simps(psd_values[idx_band], freqs[idx_band])
                            else:
                                power = 0.0
                            
                            # Store with channel and band in the name
                            feature_name = f"channel_{channel_idx}_{band_name}"
                            all_band_powers[feature_name] = power
                        
                        new_data_processed = True
                
                # If we have new data for any channel, output the band powers
                if new_data_processed and all_band_powers:
                    self.band_power_store.put(all_band_powers)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)