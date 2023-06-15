#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('data.csv')
real_distance = df.iloc[:, 1].values
measured_distance = df.iloc[:, 2].values
power = df.iloc[:, 3].values
power = 1/(power*(-1))
log_pow= np.log10(power)
delta= df.iloc[:, 4].values
ax=plt.axes(projection='3d')
ax.scatter3D(real_distance,delta,log_pow,'gray')
ax.set_xlabel('real_distance in m')
ax.set_ylabel('delta in m')
ax.set_zlabel('log_power')
plt.show()