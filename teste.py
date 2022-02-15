import pandas as pd

# create a series
s = pd.Series([-3.43, -6, 21, 6, 1.4, 'NaN'])

print(s, end='\n\n')

# calculate absolute values
result = s.abs()

#print the result
print(result)
