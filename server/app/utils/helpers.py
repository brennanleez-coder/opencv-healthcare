import numpy as np
from math import sqrt
import datetime

def calculate_angle(a, b, c):
    """
        b is the midpoint of a and c (e.g. left hip, left elbow and left shoulder)
        our case will be left-hip, left-knee and left-ankle
    """
    
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    return angle


def to_timestamp(time_float):
    return datetime.datetime.fromtimestamp(time_float).strftime('%Y-%m-%d %H:%M:%S,%f')



def calculate_distance(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
