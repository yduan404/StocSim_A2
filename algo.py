import math
import numpy as np

# --- config ---
service_mean, service_std = 1.0, 0.25
arrival_rate = 0.85
SIM_END_EVENTS = 1_000

# --- state ---
queue = []                 # FIFO of absolute arrival timestamps
current_time = 0.0         # absolute time
current_worker = None      # absolute finish timestamp (None = idle)
waiting_times = []         # realized waiting times
events_processed = 0

# --- RNG & samplers ---
rng = np.random.default_rng()

def sample_service_time():
    x = rng.normal(loc=service_mean, scale=service_std)
    while x < 0:
        x = rng.normal(loc=service_mean, scale=service_std)
    return x

def sample_interarrival():
    u = rng.random()
    return -math.log(u) / arrival_rate

# schedule first arrival
next_arrival_time = current_time + sample_interarrival()

while events_processed < SIM_END_EVENTS:
    next_finish_time = current_worker if current_worker is not None else math.inf

    if next_arrival_time <= next_finish_time:
        # process arrival
        current_time = next_arrival_time

        if current_worker is None:
            # starts service immediately
            st = sample_service_time()
            current_worker = current_time + st
            waiting_times.append(0.0)
        else:
            # join FIFO queue (store absolute arrival time)
            queue.append(current_time)

        # schedule next arrival
        next_arrival_time = current_time + sample_interarrival()

    else:
        # process completion
        current_time = next_finish_time
        current_worker = None

        if queue:
            a_time = queue.pop(0)          # FIFO
            wait = current_time - a_time   # absolute times → waiting time
            waiting_times.append(wait)
            st = sample_service_time()
            current_worker = current_time + st

    events_processed += 1

print(np.average(waiting_times))