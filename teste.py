import math
import numpy as np

def slope(x1, y1, x2, y2): # Line slope given two points:
    second = (x2-x1)
    if second > 0:
        return (y2-y1)/second
    else:
        return second/(y2-y1)
        

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

lineA = ((30, 0), (45, -1))
lineB = ((45, -1), (58, -3))

slope1 = slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

ang = angle(slope1, slope2)
print('Angle in degrees = ', abs(ang))