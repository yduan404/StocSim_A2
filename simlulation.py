import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#--------------------------------
# part 1: calculate arrival rate
#--------------------------------

airport_data = pd.read_csv('airport.csv', usecols=['Year', 'Month','Europe Passengers', 'Intercontinental Passengers', 'Total Passengers'])

sept_15_19 = airport_data[(airport_data["Year"] >= 2015)
                    & (airport_data["Year"] <= 2019)
                    & (airport_data["Month"] == "September")]

hours = 16
days = 30
lanes = 50

lambda_value = sept_15_19['Total Passengers'].mean() / (hours * days * lanes * 60)

print("=========================1=========================")
print(f"the arrival rate is {lambda_value:.3f} passgengers per minute.")

# --------------------------------
# part 2A: simulate and validate the queuing system
# --------------------------------

def get_service_time(mean, std):
    time = np.random.normal(mean, std)
    while time < 0:
        time = np.random.normal(mean, std)

    return time

def run_simulation(lambda_value, service_mean, service_std, num_servers, num_passengers, warmup_count, plot=False):
    # initialize 
    current_time = 0.0
    busy_servers = 0
    queue = []                  # list to store passenger ids waiting in line
    arrival_times = {}          # dictionary, passenger id and arrival time
    final_wait_times = []       # list to store the final wait time results
    events = []                 # event list: [time, event_type, passenger_id]       

    # counters
    served_count = 0
    passenger_count = 0    # how many passengers that we have

    # the first arrival event
    first_time = np.random.exponential(1.0 / lambda_value)
    events.append([first_time, "ARRIVAL", passenger_count])

    # main loop
    while served_count < num_passengers:
        
        # sort events by time
        events.sort(key=lambda x: x[0])

        if not events:
            break
        
        # pop the earliest event
        current_event = events.pop(0)
        
        # extract event details
        now = current_event[0]
        event_type = current_event[1]
        passenger_id = current_event[2]
        
        # update global clock
        current_time = now

        # handle arrival events
        if event_type == "ARRIVAL":
            # record the arrival time for this passenger
            arrival_times[passenger_id] = now
            
            # check if any server is free
            if busy_servers < num_servers:
                # server is free, occupy it
                busy_servers += 1
                
                # no waiting needed, 0 wait time
                final_wait_times.append(0.0)
                
                # service time for this passenger
                service_time = get_service_time(service_mean, service_std)
                depart_time = now + service_time
                events.append([depart_time, "DEPARTURE", passenger_id])
                
            else:
                # all servers are busy, add to queue
                queue.append(passenger_id)
                
            # schedule the next passenger's arrival
            passenger_count += 1
            iat = np.random.exponential(1.0 / lambda_value)
            next_arrival_time = now + iat
            events.append([next_arrival_time, "ARRIVAL", passenger_count])


        # handle departure events
        elif event_type == "DEPARTURE":
            # passenger leaves and server is free! 
            served_count += 1
            busy_servers -= 1
            
            # check if there is anyone in the queue
            if len(queue) > 0:
                # pull the next passenger from the queue
                next_passenger_id = queue.pop(0)
                
                # occupy the server again
                busy_servers += 1
                
                # calculate wait time: current time - arrival time
                arrival_t = arrival_times[next_passenger_id]
                wait = now - arrival_t
                final_wait_times.append(wait)
                
                # schedule departure for this new passenger
                service_time = get_service_time(service_mean, service_std)
                depart_time = now + service_time
                events.append([depart_time, "DEPARTURE", next_passenger_id])

    if plot:
        plt.plot(final_wait_times)
        plt.show()

    # warm-up period removal
    if len(final_wait_times) > warmup_count:
        final_wait_times = final_wait_times[warmup_count:]
        return np.mean(final_wait_times) if final_wait_times else 0.0

# testing stats
eb = 1
std = 0.25
test_utilization = 0.85
test_lambda = test_utilization / eb

run_simulation(lambda_value, 1, 0.25, 1, 3000, 1000, plot=True)

# theoretical average wait time using P-K formula 
theoretical_wait = (test_lambda * (eb ** 2 + std ** 2)) / (2 * (1 - test_utilization))

# run 40 simulations
R = np.empty(40)
for i in range(40):
    R[i] = run_simulation(test_lambda, 1, 0.25, 1, 3000, 1000)
sim_wait = np.mean(R)

# run one-sample t-test
t_stat, p_value = stats.ttest_1samp(R, theoretical_wait)


# output
print("=========================2A=========================")
print(f"theoretical average waiting time: {theoretical_wait:.3f}")
print(f"simulation average waiting time: {sim_wait:.3f}")
print(f"p-value: {p_value:.3f}")
    
if p_value > 0.05:
    print("There is no significant difference with a p-value larger than 0.05, the simulation results are consistent with the theoretical value.")
else:
    print("There is a significant difference between the simulation results and the theoretical value.")



#--------------------------------
# part 2B: evaluating interventions
#--------------------------------

base_list = np.empty(40)
a_list = np.empty(40)
b_list = np.empty(40)

for i in range(40):
    base_list[i] = run_simulation(lambda_value, 1, 0.25, 1, 3000, 0)
    a_list[i] = run_simulation(lambda_value, 1, 0.25, 2, 3000, 0)
    b_list[i] = run_simulation(lambda_value, 1, 0.1, 1, 3000, 0)

base_mean = np.mean(base_list)
a_mean = np.mean(a_list)
b_mean = np.mean(b_list)

# t-tests (two-sample)
# alternative (H_1) is less -> waiting time is reduced. 
# variance is not assumed to be equal -> using Welch's t-test
t_a, p_a = stats.ttest_ind(a_list, base_list, equal_var=False, alternative="less")
t_b, p_b = stats.ttest_ind(b_list, base_list, equal_var=False, alternative="less")

# output
print("=========================2B=========================")
print(f"baseline average waiting time: {base_mean:.3f}")
print(f"option A average waiting time: {a_mean:.3f}")
print(f"p-value of option A: {p_a:.3f}")
if p_a > 0.05:
    print("The p-value is larger than 0.05, Option A shows no statistically significant improvement.")
else:
    print("The p-value is smaller than 0.05, Option A shows statistically significant improvement.")

print(f"option B average waiting time: {b_mean:.3f}")
print(f"p-value of option B: {p_b:.3f}")
if p_b > 0.05:
    print("The p-value is larger than 0.05, Option B shows no statistically significant improvement.")
else:
    print("The p-value is smaller than 0.05, Option B shows statistically significant improvement.")

# plot A: compare between options
plt.figure(figsize=(10, 6))
plt.boxplot([base_list, a_list, b_list], tick_labels=['baseline', 'option A (2 servers)', 'option B (std=0.1)'])
plt.title('comparing waiting time of option A & B')
plt.ylabel('average waiting time')
plt.grid(True)
plt.show()

# plot B: calculate minimum servers needed
target_wait = 10  # set target waiting time as 10 minutes

server_counts = []
avg_wait_times = []
min_servers_needed = -1

for num_servers in range(1, 15):
    results = []
    for i in range(40):
        avg_wait = run_simulation(lambda_value, 1, 0.25, num_servers=num_servers, num_passengers=3000, warmup_count=0)
        results.append(avg_wait)

    final_avg = np.mean(results)
    server_counts.append(num_servers)
    avg_wait_times.append(final_avg)

    if final_avg < target_wait and min_servers_needed == -1:
        min_servers_needed = num_servers
        print(f"we need at least {num_servers} servers to keep waiting time below {target_wait:.3f} minutes")

# plotting
plt.figure(figsize=(10, 6))
plt.plot(server_counts, avg_wait_times, marker="o", linestyle="-", label="Average Wait Time")
plt.axhline(y=target_wait, color="r", label=f"target waiting time ({target_wait} min)")
if min_servers_needed != -1:
    plt.axvline(x=min_servers_needed, color='g', label=f'min servers ({min_servers_needed})')
plt.title("average waiting time vs. number of servers")
plt.xlabel("number of servers")
plt.ylabel("average waiting time in minutes")
plt.legend()
plt.grid(True)
plt.xticks(range(1, 15))
plt.show()




#--------------------------------
# bonus: adding time slot reservation
#--------------------------------

def timeslot_simulation(lambda_value, service_mean, service_std, num_servers, num_passengers, warmup_count, plot=False):
    # passengers will arrive in certain reserved time slots
    slot_minute = 5
    passenger_per_slot = int(lambda_value * slot_minute)  # the max capacity per slot
    if passenger_per_slot == 0:
        passenger_per_slot = 1  # Avoid division by zero
    num_slots = int(num_passengers / passenger_per_slot) + 1 # calculate number of slots needed

    all_arrival_times = [] # make a list for all arrival times
    for i in range(num_slots):
        start_time = i * slot_minute
        times = np.random.uniform(start_time, start_time + slot_minute, passenger_per_slot) # use uniform distribution to simulate arrivals within each slot
        all_arrival_times.extend(times)
    all_arrival_times.sort() # sort in time order
    
    
    # initialize 
    current_time = 0.0
    busy_servers = 0
    queue = []                  # list to store passenger ids waiting in line
    arrival_times = {}          # dictionary, passenger id and arrival time
    final_wait_times = []       # list to store the final wait time results
    events = []                 # event list: [time, event_type, passenger_id]       

    # counters
    served_count = 0
    passenger_count = 0    # how many passengers that we have

    # the first arrival event
    if not all_arrival_times:
        return 0.0  # No passengers to simulate

    first_time = all_arrival_times[0] # directly take from the arrival list (instead of exponential)
    events.append([first_time, "ARRIVAL", passenger_count])

    # main loop
    while served_count < num_passengers:
        
        # sort events by time
        events.sort(key=lambda x: x[0])

        if not events:
            break
        
        # pop the earliest event
        current_event = events.pop(0)
        
        # extract event details
        now = current_event[0]
        event_type = current_event[1]
        passenger_id = current_event[2]
        
        # update global clock
        current_time = now

        # handle arrival events
        if event_type == "ARRIVAL":
            # record the arrival time for this passenger
            arrival_times[passenger_id] = now
            
            # check if any server is free
            if busy_servers < num_servers:
                # server is free, occupy it
                busy_servers += 1
                
                # no waiting needed, 0 wait time
                final_wait_times.append(0.0)
                
                # service time for this passenger
                service_time = get_service_time(service_mean, service_std)
                depart_time = now + service_time
                events.append([depart_time, "DEPARTURE", passenger_id])
                
            else:
                # all servers are busy, add to queue
                queue.append(passenger_id)
                
            # schedule the next passenger's arrival
            passenger_count += 1
            if passenger_count < len(all_arrival_times):
                next_arrival_time = all_arrival_times[passenger_count]
                events.append([next_arrival_time, "ARRIVAL", passenger_count])


        # handle departure events
        elif event_type == "DEPARTURE":
            # passenger leaves and server is free! 
            served_count += 1
            busy_servers -= 1
            
            # check if there is anyone in the queue
            if len(queue) > 0:
                # pull the next passenger from the queue
                next_passenger_id = queue.pop(0)
                
                # occupy the server again
                busy_servers += 1
                
                # calculate wait time: current time - arrival time
                arrival_t = arrival_times[next_passenger_id]
                wait = now - arrival_t
                final_wait_times.append(wait)
                
                # schedule departure for this new passenger
                service_time = get_service_time(service_mean, service_std)
                depart_time = now + service_time
                events.append([depart_time, "DEPARTURE", next_passenger_id])

    # warm-up period removal
    if len(final_wait_times) > warmup_count:
        final_wait_times = final_wait_times[warmup_count:]
        return np.mean(final_wait_times) if final_wait_times else 0.0


# plot A: compare between options
# the result is not very visible, prob because lambda is too big and the time slot thing cant help too much
# so i came up with plot B testing diff lambda values <1
# dont know if we keep this one or not

num_servers_range = range(1, 11)
avg_wait_random = []
avg_wait_timeslot = []

for n_servers in num_servers_range:

    # simulate poisson arrivals
    results_random = []
    for _ in range(40):
        wait = run_simulation(lambda_value, 1, 0.25, n_servers, 3000, 0)
        results_random.append(wait)
    avg_wait_random.append(np.mean(results_random))

    # simulatie time slot arrivals
    results_timeslot = []
    for _ in range(40):
        wait = timeslot_simulation(lambda_value, 1, 0.25, n_servers, 3000, 0)
        results_timeslot.append(wait)
    avg_wait_timeslot.append(np.mean(results_timeslot))

plt.figure(figsize=(12, 7))
plt.plot(num_servers_range, avg_wait_random, marker='o', linestyle='-', label='poisson arrival')
plt.plot(num_servers_range, avg_wait_timeslot, marker='s', linestyle='--', label='time arrival')
plt.title('comparison of simulation Models vs. number of servers')
plt.xlabel('number of servers')
plt.ylabel('average waiting times')
plt.xticks(list(num_servers_range))
plt.legend()
plt.grid(True)
plt.show()


# plot B: avg waiting time vs. lambda
# the plot is to show the difference of the trend as the airport gets busier
# also what i think is important is the difference maximizes when approaching 1, the result seems nice!

lambdas = np.linspace(0.5, 0.99, 25) 

avg_wait_random = []
avg_wait_timeslot = []

for l in lambdas:
    res_r = [run_simulation(l, 1, 0.25, 1, 3000, 0) for _ in range(40)]
    avg_wait_random.append(np.mean(res_r))
    
    res_t = [timeslot_simulation(l, 1, 0.25, 1, 3000, 0) for _ in range(40)]
    avg_wait_timeslot.append(np.mean(res_t))

plt.figure(figsize=(10, 6))
plt.plot(lambdas, avg_wait_random, 'o-', label='poisson arrival')
plt.plot(lambdas, avg_wait_timeslot, 's--', label='reserved time slot arrival')

plt.title('impact of arrival rate on waiting time (server number=1, passengers=3000, E(b)=1, std=0.25)')
plt.xlabel('arrival rate (lambda)')
plt.ylabel('average waiting time')
plt.legend()
plt.grid(True)
plt.show()