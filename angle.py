# just testing stuff here

# angles will be divided into multiples of 15
# range of Yaw will be -180 to 180

def nearest_angle(theta):
	a = round(theta / 90) * 90
	if a == -180:
		return -a
	else:
		return a

# returns shortest distance between initial and target
def angle_distance(initial, target):
	a = target - initial
	if a > 180:
		return a - 360
	elif a < -180:
		return a + 360
	else:
		return a

for i in range(-180, 180):
	print(i, nearest_angle(i))