import pandas as pd
import numpy as np

# Data input
data = {
    "Year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Median_House_Price": [340000, 370000, 350000, 420000, 425000, 500000, 520000, 550000, 596000, 610000, 660000, 720000, 73000, 760000],
    "Westpac_Forecast": [56, 53, None, 13, 33, -5, 45, 34, 34, 20, -20, 15, 5, -10],
    "Joe_Bloggs_Forecast": [23, 34, 19, 42, 23, 15, 1500, 18, 19, 23, 13, 8, 7, -2],
    "Harry_Spent_Forecast": [-20, -80, -70, -80, -50, -90, -30, None, -110, -90, -60, -69, -80, -80],
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to calculate prediction error
def calculate_errors(df):
    errors = {}
    for forecaster in ["Westpac_Forecast", "Joe_Bloggs_Forecast", "Harry_Spent_Forecast"]:
        # Calculate predicted values based on the forecasts
        predicted_values = df["Median_House_Price"].shift(-1) * (1 + df[forecaster] / 100)
        
        # Calculate the absolute error and error percentage
        error = np.abs(predicted_values - df["Median_House_Price"].shift(-1)) / df["Median_House_Price"].shift(-1) * 100
        errors[forecaster] = error[:-1]  # Remove the last entry which has no forecast
    return pd.DataFrame(errors)

# Calculate errors
error_df = calculate_errors(df)

# Summary of average errors
average_errors = error_df.mean()

print("Average Prediction Errors:")
print(average_errors)

# Summary of findings
print("\nSummary of Findings:")
for forecaster, avg_error in average_errors.items():
    print(f"{forecaster}: {avg_error:.2f}% error")
