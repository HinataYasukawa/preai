import numpy as np
import json
from sklearn.linear_model import LinearRegression

num = 0
for num in range(10):
    n = f'{num:04}'
    json_open = open('frame_'+ num +'_keypoints.json')
    json_load = json.load(json_open)
    