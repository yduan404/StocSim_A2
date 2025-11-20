import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------
# Part 1: calculate arrival rate from CSV
# ---------------------------
def read_airport_data(file):
    airport_data = pd.read_csv(file, usecols=['Year', 'Month','Europe Passengers', 'Intercontinental Passengers', 'Total Passengers'])

    sept_15_19 = airport_data[
        (airport_data["Year"] >= 2015) &
        (airport_data["Year"] <= 2019) &
        (airport_data["Month"] == "September")
    ]

    hours = 16
    days = 30
    lanes = 50

    # passengers per minute per lane (Poisson rate λ per lane)
    arrival_rate = sept_15_19['Total Passengers'].sum() / (hours * days * lanes * 60)
    return arrival_rate


# ---------------------------
# Part 2: Discrete-Event simulation (one server, absolute timestamps)
# ---------------------------
class DES_sim:
    """
    One-server DES with absolute timestamps.
    - If use_target_utilization=True, overrides arrival_rate using p_target and service_time.
    - Otherwise uses the arrival_rate passed in.
    """
    def __init__(self, arrival_rate, service_time, num_servers, num_passengers, use_target_utilization=False):
        self.p_target = 0.85
        self.service_mean = float(service_time)
        self.service_std = 0.25
        self.num_passengers = num_passengers
        self.num_servers = 1
        self.event_count = 0

        if use_target_utilization:
            lambda_test = self.p_target / self.service_mean
            self.arrival_rate = float(lambda_test)
        else:
            self.arrival_rate = float(arrival_rate)

        # theoretical waiting time
        self.W_q = (self.arrival_rate * (self.service_mean**2 + self.service_std**2)) / max(1e-12, 2*(1-self.p_target))

        # state
        self.current_time = 0.0                  # absolute clock
        self.queue = []                          # FIFO: arrival timestamps waiting for service
        self.current_worker = None               # absolute finish timestamp (None = idle)
        self.waiting_times = []                  # realized waits before service

        # RNG
        self.rng = np.random.default_rng()        

    # --- sampling ---
    def _sample_service_time(self):
        x = self.rng.normal(loc=self.service_mean, scale=self.service_std)
        while x < 0:
            x = self.rng.normal(loc=self.service_mean, scale=self.service_std)
        return x

    def _sample_interarrival(self):
        u = self.rng.random()
        return -math.log(u) / self.arrival_rate

    # --- simulation main ---
    def run(self):
        next_arrival_time = self.current_time + self._sample_interarrival()
        while self.event_count < self.num_passengers:
            next_finish_time = self.current_worker if self.current_worker is not None else math.inf

            if next_arrival_time <= next_finish_time:
                # ARRIVAL event
                self.current_time = next_arrival_time

                if self.current_worker is None:
                    # starts service immediately
                    st = self._sample_service_time()
                    self.current_worker = self.current_time + st
                    self.waiting_times.append(0.0)
                else:
                    # join FIFO queue with absolute arrival timestamp
                    self.queue.append(self.current_time)

                # schedule next arrival
                next_arrival_time = self.current_time + self._sample_interarrival()

            else:
                # COMPLETION event
                self.current_time = next_finish_time
                self.current_worker = None

                # if queue non-empty, immediately start next job
                if self.queue:
                    a_time = self.queue.pop(0)              # FIFO arrival timestamp
                    wait = self.current_time - a_time
                    self.waiting_times.append(wait)
                    st = self._sample_service_time()
                    self.current_worker = self.current_time + st
            self.event_count += 1

        return self.W_q, np.array(self.waiting_times)


# ---------------------------
# Example usage (uncomment and adjust CSV path)
# ---------------------------
# csv_path = "airport.csv"
# sim = DE_sim(arrival_rate=0.85, service_time=1.0, num_servers=1, num_passengers=1000, use_target_utilization=True)
# results = sim.run()

service_time = 1.0          # mean service time
num_passengers = 100      # completions per run
num_reps = 40

lambdas = np.linspace(0.85, 2.5, 10)
mean_Wqs = []
mean_T_Wqs = []

for lam in lambdas:
    rep_means = []
    for _ in range(num_reps):
        sim = DE_sim(arrival_rate=lam,
                     service_time=service_time,
                     num_servers=1,
                     num_passengers=num_passengers,
                     use_target_utilization=False)
        T_Wq, waiting_times = sim.run()
        rep_means.append(waiting_times.mean())
    mean_Wqs.append(np.mean(rep_means))
    mean_T_Wqs.append(T_Wq)

# Plot
plt.figure()
plt.plot(lambdas, mean_Wqs, marker='o', color="red", label="Experimental W")
plt.plot(lambdas, mean_T_Wqs, marker='o', color="blue", label="Theoretical W")

plt.xlabel(r'$\lambda$ (arrival rate)')
plt.ylabel('Mean waiting time in queue')
plt.title('Mean Wq vs λ (1-server DES, 40 reps each)')
plt.legend()
plt.grid(True)
plt.show()