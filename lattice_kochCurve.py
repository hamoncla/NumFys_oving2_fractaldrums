from imports import *

def kochSquareCurveLine(start, end, level):
    if level == 0:
        return np.array([start, end])
    
    start = np.array(start)
    end = np.array(end)

    theta = np.pi / 2 
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    delta = (end - start)/ 4

    l1 = start + delta
    l2 = l1 + np.dot(R, delta)
    l3 = l2 + delta
    l4 = l3 - np.dot(R, delta)
    l5 = l4 -np.dot(R, delta)
    l6 = l5 + delta
    l7 = l6 + np.dot(R, delta)
    l8 = l7 + delta
    
    return np.vstack([kochSquareCurveLine(start, l1, level-1)[:-1],
                        kochSquareCurveLine(l1, l2, level-1)[:-1],
                        kochSquareCurveLine(l2, l3, level-1)[:-1],
                        kochSquareCurveLine(l3, l4, level-1)[:-1],
                        kochSquareCurveLine(l4, l5, level-1)[:-1],
                        kochSquareCurveLine(l5, l6, level-1)[:-1],
                        kochSquareCurveLine(l6, l7, level-1)[:-1],
                        kochSquareCurveLine(l7, l8, level-1)])

def kochSquare(L, level):
    corners = np.array([[0, 0], [L, 0], [L, L], [0, L], [0, 0]])

    result = []

    for i in range(len(corners)-1):
        start = corners[i]
        end = corners[i+1]
        side = kochSquareCurveLine(start, end, level)
        result.append(side)
    
    return np.array(result)

def createLattice(L, level, kochCurve):
    delta = L / 4**level

    allPoints = np.vstack(kochCurve)

    min_x, min_y = np.min(allPoints, axis=0) - delta
    max_x, max_y = np.max(allPoints, axis=0) + delta

    # Generate grid points around the Koch square curve
    x_vals = np.arange(min_x, max_x + delta, delta)
    y_vals = np.arange(min_y, max_y + delta, delta)

    X, Y = np.meshgrid(x_vals, y_vals)
    lattice_points = np.column_stack((X.flatten(), Y.flatten()))

    return lattice_points, x_vals, y_vals # (N, 2) array

def classification(latticePoints, kochCurve, level, distThreshold=1e-6):
    curvePoints = np.vstack(kochCurve)
    path = Path(curvePoints, closed=True)

    classification = np.zeros(len(latticePoints), dtype=int)

    insideMask = path.contains_points(latticePoints)
    classification[insideMask] = -999

    dists = cdist(latticePoints, curvePoints)
    boundary_mask = np.min(dists, axis=1) < distThreshold
    classification[boundary_mask] = -1

    insideIndices = np.where(classification == -999)[0]
    for idx, point_idx in tqdm(enumerate(insideIndices, start=1)):
        classification[point_idx] = idx

    #np.save(f"classification_matrix_level_{level}.npy", classification)

    return classification # (N,) array 1D array. Can be reshaped to (sqrt(N), sqrt(N))
