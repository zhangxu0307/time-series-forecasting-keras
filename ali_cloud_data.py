import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/machine_usage.csv", header=-1, nrows=1000000, usecols=[0, 1, 2, 3])
df.columns = ["id", "timestamp", "cpu", "mem"]
df['time'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.set_index('time', drop=True)

machine_id = 1944
metrics = "mem"

m1 = df[df["id"] == "m_"+str(machine_id)]

# print(df)

m1 = m1.resample('5T').mean().fillna(method="ffill")
m1 = m1[metrics]
print(m1)
plt.plot(m1)
plt.show()

m1.to_csv("m_"+str(machine_id)+"_"+metrics+".csv", header=[metrics])

# cpu: 1935, 1944, 1955
# mem: 1935, 1944, 1955