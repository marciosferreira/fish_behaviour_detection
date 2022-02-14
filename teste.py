import math

def slope(x1, y1, x2, y2): # Line slope given two points:
    print(x1)
    print(x2)
    return (y2-y1)/(x2-x1)


def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

lineA = ((411, 797), (402, 801))
lineB = ((411, 797), (411, 785))

#((411, 797), (402, 801))
#((411, 797), (411, 785))


dif1 = abs(lineA[0][0] - lineA[1][0])
dif2 = abs(lineA[0][1] - lineA[1][1])
if min(dif1, dif2) == dif1:
    print('11111111')
    slope1 = slope(lineA[0][1], lineA[0][0], lineA[1][1], lineA[1][0])
    slope2 = slope(lineB[0][1], lineB[0][0], lineB[1][1], lineB[1][0])
else:
    print('2222222222')
    slope1 = slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
    slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

ang = angle(slope1, slope2)
print('Angle in degrees = ', ang)

