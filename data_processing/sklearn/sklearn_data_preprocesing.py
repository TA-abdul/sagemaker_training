import numpy as np
import pandas as pd

input_data_path = "insurance.csv"
df = pd.read_csv(input_data_path)


#First, we need to import the scale from preprocessing that belongs to sklearn
from sklearn.preprocessing import scale
charges_scaled = scale(df.charges)

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Create a new transformer. If you do not set any paramter, the data is scaled to the range [0, 1]. 
# in order to set a range, you need to include: feature_range=(min, max). In the example we use min=3, max=7
my_created_transformer = MinMaxScaler(feature_range=(3, 7))
# Fit the data to scaler
my_created_transformer.fit(df.charges.values.reshape(-1, 1))
#Transform the data
charges_scaled_range = my_created_transformer.transform(df.charges.values.reshape(-1, 1))

df['charges_scaled_range'] = charges_scaled_range

output_data_path = "processed_insurance.csv"
df.to_csv(output_data_path)
