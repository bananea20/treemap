import pandas as pd
import matplotlib.pyplot as plt
import squarify_modify  as  squarify  
import random



def generate_group_list(n):
    group_list = []
    for i in range(n):
        group_list.append("group " + chr(ord('A') + i))
    return group_list

# 用法示例


def generate_weight_list(n, max_weight):
    weight_list = []
    for _ in range(n):
        weight_list.append(random.randint(1, max_weight))
    return weight_list

# 用法示例
# num_weights = 10
# max_weight = 10
# result = generate_weight_list(num_weights, max_weight)
# print(result)
