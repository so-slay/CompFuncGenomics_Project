import pandas as pd
import numpy as np

from collections import Counter

data = ['abhfhvjnvadfjjjjjjjjgjjgj']

counts = Counter(data[0])
# Alternate way to count:
alt = data.count('a')




print(counts)