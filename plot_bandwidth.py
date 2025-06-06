import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Read the CSV file
df = pd.read_csv('bandwidth_data.csv')

# Convert timestamp to datetime objects
base_time = datetime.strptime('00:00:00', '%H:%M:%S')
df['Time'] = df['Timestamp'].apply(lambda x: base_time + timedelta(
    hours=int(x.split(':')[0]),
    minutes=int(x.split(':')[1]),
    seconds=int(x.split(':')[2])
))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot all three bandwidth metrics
plt.plot(df['Time'], df['Read_Bandwidth_MB_s'], label='Read Bandwidth', color='blue')
plt.plot(df['Time'], df['Write_Bandwidth_MB_s'], label='Write Bandwidth', color='green')
plt.plot(df['Time'], df['Total_Bandwidth_MB_s'], label='Total Bandwidth', color='red')

# Format the plot
plt.xlabel('Time (HH:MM:SS)')
plt.ylabel('Bandwidth (MB/s)')
plt.title('Memory Bandwidth Over Time')
plt.grid(True)
plt.legend()

# Format x-axis to show time more clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# Ensure no overlapping of x-labels
plt.tight_layout()

# Save the plot
plt.savefig('bandwidth_plot.png', dpi=300)

# Display the plot
plt.show() 