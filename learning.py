import numpy as np
import json
from sklearn.linear_model import LinearRegression

num = 0
for num in range(10):
    n = f'{num:04}'
    json_open = open('frame_'+ num +'_keypoints.json')
    data = json.load(json_open)
    
    indices = [0,4,7]
    keypoint_names = ['face', 'right', 'left']
    
    for person in data['people']:
        keypoints = person['pose_keypoints_2d']

        for index, name in zip(indices, keypoint_names):
            x = keypoints[index * 3]  # x座標
            y = keypoints[index * 3 + 1]  # y座標
            confidence = keypoints[index * 3 + 2]  # 信頼度
            print(f"{name}: x = {x}, y = {y}, 信頼度 = {confidence}")