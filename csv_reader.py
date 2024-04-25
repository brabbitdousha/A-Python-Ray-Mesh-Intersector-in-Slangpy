import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
import numpy as np

def str_to_float(x):
    try:
        return float(x)
    except ValueError:
        return x
    
df = pd.read_csv('D:/workspace/BVHVisualization/resources/bvhexample/origin.csv', delimiter='\s+')
#df = pd.read_csv('./data.csv', delimiter='\s+')
left_array = df['left'].values
right_array = df['right'].values
primitiveIdx_array = df['primitiveIdx'].values
aabb_min_x_array = df['aabb_min_x'].values
aabb_min_y_array = df['aabb_min_y'].values
aabb_min_z_array = df['aabb_min_z'].values
aabb_max_x_array = df['aabb_max_x'].values
aabb_max_y_array = df['aabb_max_y'].values
aabb_max_z_array = df['aabb_max_z'].values

print(np.where(aabb_min_x_array == 0)[0])
print(np.where(aabb_min_y_array == 0)[0])
print(np.where(aabb_min_z_array == 0)[0])
print(np.where(aabb_max_x_array == 0)[0])
print(np.where(aabb_max_y_array == 0)[0])
print(np.where(aabb_max_z_array == 0)[0])


