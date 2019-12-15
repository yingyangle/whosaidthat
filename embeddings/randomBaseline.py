# Unused - can be deleted
# Duanchen Liu
# Generate random baselines for the datasets
import random
import csv
import pandas as pd
import numpy as np

dataset_name = "bang.csv"
# print(len(dataset_name))
# print(random.randint(0, 6))

data = pd.read_csv(dataset_name) # read in csv
instance_num = len(data) # get the length of csv
generated_array = []
for i in range(instance_num):
    generated_array.append(random.randint(0, 6))


print(generated_array)

