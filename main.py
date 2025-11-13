import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# part 1: calculate arrival rate

airport_data = pd.read_csv('airport.csv', usecols=['Year', 'Month','Europe Passengers', 'Intercontinental Passengers', 'Total Passengers'])

sept_15_19 = airport_data[(airport_data["Year"] >= 2015)
                    & (airport_data["Year"] <= 2019)
                    & (airport_data["Month"] == "September")]

hours = 16
days = 30
lanes = 50

arrival_rate = sept_15_19['Total Passengers'].sum() / (hours * days * lanes * 60)
print(arrival_rate)
