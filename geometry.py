import numpy as np
from math import sqrt, pi
from numba import njit
from shapely import LineString, Point, Polygon
import shapely


def center(vertices):
	n = len(vertices)
	sumX = 0
	sumY = 0
	# sum the vectors
	for i in range(0,n):
		x,y = vertices[i]
		sumX += x
		sumY += y

	# divide by number of sides
	cx = sumX / n
	cy = sumY / n

	return cx,cy


@njit
def periodic_diff(v1, v2, L):
    return ((v1 - v2 + L / 2.0) % L) - L / 2.0


@njit
def euclidean_distance(x0, y0, x1, y1):
    return sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


@njit
def unit_vector(v1, v2):
    vector = v1 - v2
    dist = euclidean_distance(v1[0], v1[1], v2[0], v2[1])
    if dist < 1e-6:
        return np.array([0.0, 0.0])
    return vector / dist


@njit
def cross_product_periodic(v1, v2, L):
	x1 = float(v1[0])
	y1 = float(v1[1])
	x2 = float(v2[0])
	y2 = float(v2[1])
	Lx = float(L[0])
	Ly = float(L[1])

	# Adjust for periodic boundary conditions
	diff_x = x2 - x1
	diff_y = y2 - y1
	dx = diff_x - Lx * round(diff_x / Lx)
	dy = diff_y - Ly * round(diff_y / Ly)

	# Calculate the cross product
	cross_product = x1 * dy - y1 * dx

	return cross_product


@njit
def area_periodic(vertices, L):

	area = 0.0

	for i in range(len(vertices)):

		v1 = vertices[i]
		v2 = vertices[(i + 1) % len(vertices)]
		cross_product = cross_product_periodic(v1, v2, L)
		area += cross_product
	
	return 0.5 * abs(area)


@njit
def perimeter(vertices):
	n = len(vertices)
	perimeter = 0.
	for i in range(0,n):
		x0,y0 = vertices[i]
		if i == n - 1:
			x1,y1 = vertices[0]
		if i != n - 1:
			x1,y1 = vertices[i+1]
		dist = euclidean_distance(x0, y0, x1, y1)
		perimeter += dist
	return perimeter


def rand_angle():
    return np.random.uniform(-pi, pi)


def tri_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = tri_area(x1, y1, x2, y2, x3, y3)
    A1 = tri_area(x, y, x2, y2, x3, y3)
    A2 = tri_area(x1, y1, x, y, x3, y3)
    A3 = tri_area(x1, y1, x2, y2, x, y)

    if (A - (A1 + A2 + A3)) < 0.0001:
        return True
    return False


def CrossProduct(A):
    x1 = A[1][0] - A[0][0]
    y1 = A[1][1] - A[0][1]
    x2 = A[2][0] - A[0][0]
    y2 = A[2][1] - A[0][1]

    return (x1 * y2 - y1 * x2)


def isConvex(points):
    n = len(points)
    prev = 0
    curr = 0

    for i in range(n):
        temp = [points[i], points[(i + 1) % n], points[(i + 2) % n]]
        curr = CrossProduct(temp)

        if curr != 0:
            if curr * prev < 0:
                return False
            prev = curr

    return True


@njit(cache=True)
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_to_segment_distance(px, py, ax, ay, bx, by):
    segment_length_squared = (bx - ax) ** 2 + (by - ay) ** 2
    if segment_length_squared < 1e-12 or np.isnan(segment_length_squared):
        print("segment_length_squared: ", segment_length_squared)
        print("ax=", ax, " ay=", ay)
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2), (ax, ay)

    t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / segment_length_squared))
    projection_x = ax + t * (bx - ax)
    projection_y = ay + t * (by - ay)
    distance = np.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)

    if np.isnan(np.array([projection_x, projection_y])).any():
        print("NaN in point_to_segment_distance")
        print("px=", px, " py=", py)
        print("distance=", distance)
        print("segment_length_squared=", segment_length_squared)
        print("t=", t)

    return distance, (projection_x, projection_y)


def move_point_to_nearest_edge(point, polygon):
    if not is_point_in_polygon(point, polygon):
        return point

    min_distance = float("inf")
    nearest_point = point

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        distance, projection = point_to_segment_distance(point[0], point[1], p1[0], p1[1], p2[0], p2[1])
        if distance < min_distance:
            min_distance = distance
            nearest_point = projection

    return nearest_point


def check_inside(point, polygon):
    if len(polygon) < 3:
        return False

    poly = Polygon(polygon)
    pt = Point(point)

    return shapely.within(pt, poly)


def move_to_intersection_point(point1, point2, polygon):
    if not is_point_in_polygon(point1, polygon):
        return point1

    line = LineString([point1, point2])

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edge = LineString([p1, p2])

        if line.intersects(edge):
            intersection = line.intersection(edge)
            return (intersection.x, intersection.y)

    return point1


def get_intersection_edge(point1, point2, polygon):
    line = LineString([point1, point2])

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edge = LineString([p1, p2])

        if line.intersects(edge):
            intersection = line.intersection(edge)
            return [i, (i + 1) % len(polygon)]

    return None

def nematic_director(vertices):
	
	xC,yC = center(vertices)
	n = len(vertices)
	v = np.array(vertices)
	G = np.zeros((2,2))
	for i in range(0,n):
		x,y = vertices[i]
		r   = np.atleast_2d(np.array([x-xC,y-yC]).T)
		# print("G'=",r*r.T)
		G  += r*r.T 
	G = G/n 
	if np.isnan(G).any():
		print("G is nan")
		print('xC=',xC,' yC=',yC)
		print('n:',n)
		print('vertices:',vertices)
		print('G:',G)
		

	eval, evec = np.linalg.eig(G)

	if 0.95*eval[0] > eval[1]:
		nd = evec[:,0]
	elif 0.95*eval[1] > eval[0]:
		nd = evec[:,1]
	else:
		nd = np.array([0,0]).T

	# neg = np.random.choice([-1,1])

	return nd,eval 