import pandas as pd
import math
import numpy as np
one= [float(x) for x in range(1,201)]
two= [x+2 + np.random.normal(0,1) for x in range(1,201)]
df = pd.DataFrame({'one':one,'two':two})
print(df)
df.to_csv('data.csv',index=False)
