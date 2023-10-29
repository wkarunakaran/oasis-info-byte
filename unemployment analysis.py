import pandas as pd
import matplotlib.pyplot as plt

# Load your unemployment rate data (assuming you have a CSV file with the provided column names)
data = pd.read_csv("C:\\Users\\karan\\Documents\\Python Scripts\\Unemployment in India (1).csv")

# Convert the ' Date' column to a datetime object (notice the space before 'Date')
data[' Date'] = pd.to_datetime(data[' Date'])

# Sort the data by date (optional but recommended)
data = data.sort_values(by=' Date')

# Plot the unemployment rate over time
plt.figure(figsize=(12, 6))
plt.plot(data[' Date'], data[' Estimated Unemployment Rate (%)'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate Over Time')
plt.grid(True)
plt.show()

   
