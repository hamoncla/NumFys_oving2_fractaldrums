@jit(nopython=True, cache=True)
def checkPointEvenOdd(point, kochCurve):
    xPoint, yPoint = point
    crossings = 0  # Count boundary crossings

    for side in kochCurve:
        xKoch1, yKoch1 = side[:-1, 0], side[:-1, 1]
        xKoch2, yKoch2 = side[1:, 0], side[1:, 1]

        for (x1, y1), (x2, y2) in zip(zip(xKoch1, yKoch1), zip(xKoch2, yKoch2)):
            if np.isclose(xPoint, x1) and np.isclose(yPoint, y1):
                return "boundary"
            if np.isclose(xPoint, x2) and np.isclose(yPoint, y2):
                return "boundary"
        
        # Count crossings (ray-casting algorithm)
        for (x1, y1), (x2, y2) in zip(zip(xKoch1, yKoch1), zip(xKoch2, yKoch2)):
            if (y1 > yPoint) != (y2 > yPoint):  
                x_intersect = x1  # Find the intersection point
                
                if x_intersect > xPoint:  # Only consider crossings to the right of the point
                    crossings += 1

    # Odd number of crossings → inside, even number → outside
    return "inside " if crossings % 2 == 1 else "outside"

def checkPointWindingNumber(point, kochCurve):
    return 0