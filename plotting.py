import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plotKochLineCurve(levels = [0, 1, 2]):
    from solve import kochSquareCurveLine
    start = [0, 0]
    end = [1, 0]
    annotations = ['a)', 'b)', 'c)']

    fig, axs = plt.subplots(1, len(levels), figsize=(12, 4))
    for i, level in enumerate(levels):
        curve = kochSquareCurveLine(start, end, level)
        axs[i].plot(curve[:, 0], curve[:, 1], color='blue')
        axs[i].set_title(f'Generation {level}', fontsize=15)
        axs[i].annotate(annotations[i], xy=(0.02, 0.90), xycoords='axes fraction', fontsize=25)

    fig.tight_layout()
    #plt.show()
    fig.savefig('figures_koch_lattice/koch_line_curve_levels_0_to_2.png', dpi=300)    

def plotKochSquares(L):
    from solve import kochSquare

    annotations = ['a)', 'b)', 'c)', 'd)']
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    levels = [0, 1, 2, 3]
    for i, ax in enumerate(axs.flatten()):
        sides = kochSquare(L, levels[i])
        for side in sides:
            ax.plot(side[:, 0], side[:, 1], color='blue')
        ax.set_title(f'Generation {levels[i]}', fontsize=15)
        ax.set_aspect('equal')
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.annotate(annotations[i], xy=(0.06, 0.85), xycoords='axes fraction', fontsize=22)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'figures_koch_lattice/koch_square_levels_0_to_3_wAnn_woBoxes.png', dpi=300)
    #plt.show()

def plotKochSquare(kochCurve):
    for side in kochCurve:
        plt.plot(side[:, 0], side[:, 1], color='black', linewidth=1)
    #plt.axis('off')
    #plt.show()

def plotKochSquareLattice(kochCurve, lattice, method='fivepoint'):
    fig = plt.figure(figsize=(6, 5.6))
    for side in kochCurve:
        plt.plot(side[:, 0], side[:, 1], color='blue')
    plt.plot(lattice[:,0], lattice[:,1], 'o', markersize = 4, color = 'red', alpha = 0.5)
    plt.axis('equal')
    plt.title(r"Lattice, $\ell = 2$", fontsize=18)
    plt.axis('off')
    fig.tight_layout()
    #plt.show()
    plt.savefig(f'figures_{method}/koch_square_lattice.png', dpi=300)

def testMatrixPlot(lattice):
    plt.figure(figsize=(8, 8))

    # --- Colormap setup ---
    colors = ['red', 'green']  # Boundary (-1), Outside (0)
    cmap_main = plt.cm.Blues  # For inside (>0)

    inside_values = lattice[lattice > 0]
    max_inside = np.max(inside_values) if inside_values.size > 0 else 1
    norm_inside = plt.Normalize(vmin=1, vmax=max_inside)  

    cmap_list = [mcolors.to_rgba('red'), mcolors.to_rgba('green')]
    num_inside_colors = 256
    cmap_inside = cmap_main(np.linspace(0.3, 1, num_inside_colors))
    full_cmap = mcolors.ListedColormap(cmap_list + list(cmap_inside))

    bounds = [-1.5, -0.5, 0.5] + list(np.linspace(1, max_inside, num_inside_colors))
    norm = mcolors.BoundaryNorm(bounds, full_cmap.N)

    # --- Matrix plot ---
    plt.imshow(lattice, cmap=full_cmap, norm=norm, interpolation='nearest', origin='upper')

    # --- Colorbar ---
    cbar = plt.colorbar()
    cbar.set_ticks([-1, 0, max_inside])
    cbar.set_label("Lattice Values")

    # --- Labels ---
    plt.title("Lattice Visualization with Koch Square Overlay")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(False)
    plt.show()

def plotClassifiedLattice(lattice_points, classifications, kochCurve):
    plt.figure(figsize=(8, 8))

    # Split points based on classification
    outside = lattice_points[classifications == 0]
    boundary = lattice_points[classifications == -1]
    inside_mask = classifications > 0
    inside = lattice_points[inside_mask]
    inside_values = classifications[inside_mask]

    # Plot Koch square curve
    for side in kochCurve:
        plt.plot(side[:, 0], side[:, 1], color='black', linewidth=1)

    # Plot outside points (green)
    if len(outside) > 0:
        plt.plot(outside[:, 0], outside[:, 1], 'o', color='green', markersize=3, label='Outside (0)')

    # Plot boundary points (red)
    if len(boundary) > 0:
        plt.plot(boundary[:, 0], boundary[:, 1], 'o', color='red', markersize=3, label='Boundary (-1)')

    # Plot inside points with color gradient
    if len(inside) > 0:
        norm = plt.Normalize(vmin=1, vmax=np.max(inside_values))
        cmap = plt.cm.Blues

        # Use a custom colormap range to make the first blue values more intense
        adjusted_cmap = cmap(np.linspace(0.5, 1, 256))  # Shift to avoid light blues
        adjusted_cmap = plt.cm.colors.ListedColormap(adjusted_cmap)

        plt.scatter(inside[:, 0], inside[:, 1], c=inside_values, cmap=adjusted_cmap, norm=norm, s=10, label='Inside (>0)')

    # Final touches
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.title("Lattice Classification with Koch Curve Overlay")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def plotEigenmodes(eigenVectors, eigenValues, classified, x_vals, y_vals, 
                   mode_indices, level, numEigVals, method, kochCurve=None, cmap='seismic', save=False):

    from solve import mapEigenvectors

    X, Y = np.meshgrid(x_vals, y_vals)
    n_modes = len(mode_indices)

    n_rows = 2
    n_cols = int(np.ceil(n_modes / 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8)) # 12x6 for 10 modes
    axes = axes.flatten()

    for i, mode_idx in enumerate(mode_indices):
        eigenVec = eigenVectors[:, mode_idx]
        eigenLattice = mapEigenvectors(classified, eigenVec)

        vmax = np.max(np.abs(eigenLattice))
        vmin = -vmax

        im = axes[i].contourf(X, Y, eigenLattice, cmap=cmap, levels=100, vmin=vmin, vmax=vmax)
        axes[i].set_title(f"λ = {eigenValues[i].round(4)}", fontsize=18)
        axes[i].set_aspect("equal")
        axes[i].axis("off")

        annotation = f"{chr(97 + i)})"  # chr(97) is 'a'
        axes[i].text(0.02, 0.95, annotation, transform=axes[i].transAxes,
                     fontsize=20, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Plot the Koch curve
        if kochCurve is not None:
            curvePoints = np.vstack(kochCurve)
            axes[i].plot(curvePoints[:, 0], curvePoints[:, 1], color='black', linewidth=0.5)
        

    fig.tight_layout()
    if save:
        plt.savefig(f'figures_{method}/eigenmodes_level_{level}_modes_{len(mode_indices)}2.png', dpi=300)
    else:
        plt.show()

def plotDeltaN(eigenFrequencies, level):
    from solve import calculateNOmega
    from scipy.stats import linregress

    omegaSquared = eigenFrequencies**2

    NOmega = calculateNOmega(eigenFrequencies)

    deltaN = omegaSquared/(4*np.pi) - NOmega

    logDeltaN = np.log(deltaN)
    logOmega = np.log(eigenFrequencies)

    

    linReg = linregress(logOmega, logDeltaN)
    d = linReg.slope
    intercept = linReg.intercept

    plt.scatter(logOmega, logDeltaN, s=7, alpha=0.5, label="Data points")
    plt.plot(logOmega, d*logOmega + intercept, color='red', linestyle='--', label=f"d = {d:.4f}")
    plt.legend(fontsize=13)
    plt.xlabel(r"$\log(\omega)$", fontsize=15)
    plt.ylabel(r"$\log(\Delta N)$", fontsize=15)
    plt.title(f"Level {level}", fontsize = 20)
    plt.show()
    
    #plt.savefig(f"results_{method}/deltaN_level_{level}.png", dpi = 300)

def plotDeltaNMultipleLevels(levels, numEigVals, method):
    from solve import calculateNOmega
    from scipy.stats import linregress

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    annotations = ['a)', 'b)']

    for i, level in enumerate(levels):
        ax = axs[i]

        eigenFrequencies = np.loadtxt(f"results_{method}/eigenvalues_level_{level}_{numEigVals}.txt").round(9)
        omegaSquared = eigenFrequencies**2

        NOmega = calculateNOmega(eigenFrequencies)

        deltaN = omegaSquared/(4*np.pi) - NOmega

        logDeltaN = np.log(deltaN)
        logOmega = np.log(eigenFrequencies)

        

        linReg = linregress(logOmega, logDeltaN)
        d = linReg.slope
        intercept = linReg.intercept

        ax.scatter(logOmega, logDeltaN, s=7, alpha=0.5, label="Data points")
        ax.plot(logOmega, d*logOmega + intercept, color='red', linestyle='--', label=f"d = {d:.4f}")
        ax.legend(fontsize=18)
        # ax.set_xlabel(r"$\log(\omega)$", fontsize=15)
        # ax.set_ylabel(r"$\log(\Delta N)$", fontsize=15)
        ax.set_title(f"Level {level}", fontsize = 22)
        ax.annotate(annotations[i], xy=(0.015, 0.92), xycoords='axes fraction', fontsize=30)

    fig.supxlabel(r"$\log(\omega)$", fontsize=20)
    fig.supylabel(r"$\log(\Delta N)$", fontsize=20)
    fig.tight_layout()
    #plt.show()
    fig.savefig(f"results_{method}/deltaN_levels_{levels[0]}_{levels[1]}.png", dpi = 300)







