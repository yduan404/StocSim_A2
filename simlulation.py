import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# part 1: calculate arrival rate

airport_data = pd.read_csv('airport.csv', usecols=['Year', 'Month','Europe Passengers', 'Intercontinental Passengers', 'Total Passengers'])

sept_15_19 = airport_data[(airport_data["Year"] >= 2015)
                    & (airport_data["Year"] <= 2019)
                    & (airport_data["Month"] == "September")]

hours = 16
days = 30
lanes = 50

arrival_rate = sept_15_19['Total Passengers'].mean() / (hours * days * lanes * 60)
lambda_value = arrival_rate


def get_service_time(mean, std):
    time = np.random.normal(mean, std)
    while time < 0:
        time = np.random.normal(mean, std)

    return time

def run_simulation(lambda_value, service_mean, service_std, num_servers, num_passengers):
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
    #print(final_wait_times)
    print("average wait time: ", np.mean(final_wait_times))
    return np.mean(final_wait_times)

# testing stats
eb = 1
std = 0.25
test_utilization = 0.85
test_lambda = test_utilization / eb
theoretical_wait = (test_lambda * (eb ** 2 + std ** 2)) / (2 * (1 - test_utilization))
print("theoretical average wait time: ", theoretical_wait)

R = np.empty(40)
for i in range(40):
    R[i] = run_simulation(test_lambda, 1, 0.25, 1, 3000)

print(np.mean(R))
