# 1. Load your dataset
import pandas as pd
df = pd.read_csv('cleaned_H2.csv')



# ---------------------------------------------------------

# SCENARIO A: BOTH columns must not be 0 (AND condition)

# ---------------------------------------------------------



# ---------------------------------------------------------

# SCENARIO B: AT LEAST ONE column must not be 0 (OR condition)

# ---------------------------------------------------------

filtered_either = df[(df['PreviousBookingsNotCanceled'] != 0)]



print("\nRows where AT LEAST ONE column is not zero:")

print(filtered_either[['IsRepeatedGuest', 'PreviousCancellations','RequiredCarParkingSpaces']].head())



filtered_either = df[(df['IsRepeatedGuest'] != 0) | (df['PreviousCancellations'] != 0) |(df['RequiredCarParkingSpaces'] != 0)]



print("\nRows where AT LEAST ONE column is not zero:")

print(filtered_either[['IsRepeatedGuest', 'PreviousCancellations','RequiredCarParkingSpaces']])
