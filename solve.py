from imports import *
from lattice_kochCurve import *
from eigen_matrices import *
from plotting import *


def createEigenMatrixFivePoint(classifiedMatrix, delta, level, L):
    indicesInside = np.argwhere(classifiedMatrix > 0)
    N = len(indicesInside)

    insideIndexMap = {tuple(idx): i for i, idx in enumerate(indicesInside)}

    rows, cols, data = [], [], []

    for idx, (i, j) in tqdm(enumerate(indicesInside), total=N):
        rows.append(idx)
        cols.append(idx)
        data.append((4*L**2)/(delta**2))

        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbour = (i+di, j+dj)

            if neighbour in insideIndexMap:
                rows.append(idx)
                cols.append(insideIndexMap[neighbour])
                data.append(-L**2/(delta**2))

    print(len(data), len(rows), len(cols))

    print(N)

    eigenMatrix = csr.csr_matrix((data, (rows, cols)), shape=(N, N))

    return eigenMatrix

def createEigenMatrixNinePoint(classifiedMatrix, delta, level, L):
    indicesInside = np.argwhere(classifiedMatrix > 0)
    N = len(indicesInside)

    insideIndexMap = {tuple(idx): i for i, idx in enumerate(indicesInside)}

    matrix = lil_matrix((N, N), dtype=float)

    centerCoeff = (20*L**2)/(6*delta**2)
    offDiagonalCoeff = (-4*L**2)/(6*delta**2)
    diagonalCoeff = (-1*L**2)/(6*delta**2)

    for centerIdx, (i, j) in tqdm(enumerate(indicesInside), total=N):
        matrix[centerIdx, centerIdx] = centerCoeff

        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]: # Off-diagonal neighbours
            neighbour = (i+di, j+dj)
    
            if neighbour in insideIndexMap:
                neighbourIdx = insideIndexMap[neighbour]
                matrix[centerIdx, neighbourIdx] = offDiagonalCoeff
        

        for di, dj in [(-1,-1), (-1,1), (1,-1), (1,1)]: # Diagonal neighbours
            neighbour = (i+di, j+dj)

            if neighbour in insideIndexMap:
                neighbourIdx = insideIndexMap[neighbour]
                matrix[centerIdx, neighbourIdx] = diagonalCoeff

    return matrix.tocsr()

def createEigenMatrixBiharmonic(classifiedMatrix, delta, level, L):
    indicesInside = np.argwhere(classifiedMatrix > 0)

    insideIndexMap = {tuple(idx): i for i, idx in enumerate(indicesInside)}

    ghostMap = {}
    ghosts = 0

    row, col, data = [], [], []

    centerCoeff = 20/(delta**4)
    offDiagonalCoeffClosest = -8/(delta**4)
    diagonalCoeff = 2/(delta**4)
    offDiagonalCoeffFarthest = 1/(delta**4)

    for centerIdx, (i, j) in tqdm(enumerate(indicesInside), total=len(indicesInside)):
        row.append(centerIdx)
        col.append(centerIdx)
        data.append(centerCoeff)

        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]: # Off-diagonal neighbours
            neighbour = (i+di, j+dj)
            if neighbour in insideIndexMap:
                neighbourIdx = insideIndexMap[neighbour]
                row.append(centerIdx)
                col.append(neighbourIdx)
                data.append(offDiagonalCoeffClosest)
        
        for di, dj in [(-1,-1), (-1,1), (1,-1), (1,1)]: # Diagonal neighbours
            neighbour = (i+di, j+dj)
            if neighbour in insideIndexMap:
                neighbourIdx = insideIndexMap[neighbour]
                row.append(centerIdx)
                col.append(neighbourIdx)
                data.append(diagonalCoeff)
        
        for di, dj in [(-2,0), (2,0), (0,-2), (0,2)]: # Far away neighbours
            neighbour = (i+di, j+dj)
            if neighbour in insideIndexMap:
                neighbourIdx = insideIndexMap[neighbour]
                row.append(centerIdx)
                col.append(neighbourIdx)
                data.append(offDiagonalCoeffFarthest)

            elif classifiedMatrix[neighbour] == 0:
                if neighbour not in ghostMap:    
                    ghostMap[neighbour] = len(indicesInside) + ghosts
                    ghosts += 1

                ghostIdx = ghostMap[neighbour]

                row.append(centerIdx)
                col.append(ghostIdx)
                data.append(centerCoeff)

    N = len(indicesInside) + ghosts
    
    matrix = csr.csr_matrix((data, (row, col)), shape=(N, N))

    return matrix

def solveEigenProblem(matrix, numEigVals, level, method, save=False):
    if method == "biharmonic" and save == False:
        eigenValues, eigenVectors = sparse.eigs(matrix, numEigVals, which='SM') # landa = (omega/v * L)^2
        return eigenValues, eigenVectors
    
    elif method == "biharmonic" and save == True:
        eigenValues, eigenVectors = sparse.eigs(matrix, numEigVals, which='SM') # landa = (omega/v * L)^2

        np.savetxt(f"results_{method}/eigenvalues_level_{level}_{numEigVals}.txt", eigenValues)
        np.save(f"results_{method}/eigenvectors_level_{level}_{numEigVals}.npy", eigenVectors)
        
        return eigenValues, eigenVectors

    elif method != "biharmonic" and save == True:
        eigenValues, eigenVectors = sparse.eigsh(matrix, numEigVals, which='SM') # landa = (omega/v * L)^2
        eigenFrequencies = np.sqrt(eigenValues) # Big Omega = sqrt(landa)

        np.savetxt(f"results_{method}/eigenvalues_level_{level}_{numEigVals}.txt", eigenFrequencies)
        np.save(f"results_{method}/eigenvectors_level_{level}_{numEigVals}.npy", eigenVectors)
    
        return eigenFrequencies, eigenVectors
    
    elif method != "biharmonic" and save == False:
        eigenValues, eigenVectors = sparse.eigsh(matrix, numEigVals, which='SM')
        eigenFrequencies = np.sqrt(eigenValues)
        return eigenFrequencies, eigenVectors

def mapEigenvectors(classified, eigenVec):
    insideIndices = np.argwhere(classified > 0).flatten()
    
    eigenLattice = np.zeros(len(classified))

    N = int(np.sqrt(len(classified)))

    for i, idx in enumerate(insideIndices):
        eigenLattice[idx] = eigenVec[i]

    return eigenLattice.reshape(N, N)

# Eigenvalues in file = omega/v * L
# Delta(N) scales with omega^d

def calculateNOmega(eigenFrequencies):
    NOmega = np.array([np.sum(eigenFrequencies <= w) for w in eigenFrequencies])
    return NOmega

def createEigenMatrix(method, classifiedMatrix, delta, level, L):
    if method == "fivepoint":
        eigenMatrix = createEigenMatrixFivePoint(classifiedMatrix, delta, level, L)
    elif method == "ninepoint":
        eigenMatrix = createEigenMatrixNinePoint(classifiedMatrix, delta, level, L)
    elif method == "biharmonic":
        eigenMatrix = createEigenMatrixBiharmonic(classifiedMatrix, delta, level, L)

    return eigenMatrix

if __name__ == "__main__":
    # Parameters
    level = 4
    L = 1
    delta = L / 4**level
    numEigVals = 10
    method = "biharmonic"

    # Stuff for plotting
    mode_indices = list(range(8))

    # Create koch curve and lattice points
    kochCurve = kochSquare(L, level)
    latticePoints, x_vals, y_vals = createLattice(L, level, kochCurve)

    classified = np.load(f"classification_matrices/classification_matrix_level_{level}.npy")
    N = int(np.sqrt(len(classified)))
    classifiedMatrix = classified.reshape((N, N))

    # Create eigenmatrix
    #eigenMatrix = createEigenMatrix(method, classifiedMatrix, delta, level, L)

    # # Solve eigenproblem

    # # Solve eigenproblem
    #eigenValues, eigenVectors = solveEigenProblem(eigenMatrix, numEigVals, level, method, save=False)
    # Load stuff

    eigenValues = np.loadtxt(f"results_{method}/eigenvalues_level_{level}_{numEigVals}.txt", dtype = complex).real

    eigenVectors = np.load(f"results_{method}/eigenvectors_level_{level}_{numEigVals}.npy")

    eigenValues = eigenValues[:8]

    eigenVectors = eigenVectors[:, :8]

    print(eigenVectors.shape)

    plotEigenmodes(eigenVectors, eigenValues, classified, x_vals, y_vals, mode_indices, level, numEigVals, method, kochCurve=kochCurve, save=True)










