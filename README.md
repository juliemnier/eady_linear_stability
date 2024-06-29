# Eady Linear Stability

These Python scripts solve the linear stability problem for the Eady problem in quasi-geostrophy. The equations implemented follow the same syntax as in [this repository](https://github.com/BenMql/coral) (using Chebyshev polynomials).

---

## Scripts

### `driver_eadyQG`
- **Description**: Returns the spectrum, eigenvalues, and eigenvectors for the Eady problem.
- **Features**: Plots the analytical solution's vertical profiles for the linear stability problem as stated in G.K. Vallis' *Atmospheric and Oceanic Fluid Dynamics*, Second Edition, in the f-plane approximation.

### `driver_eadybetaQG`
- **Description**: Compares the plots from the f-plane approximation problem with the beta problem, accounting for the Earth's curvature.
- **Note**: Results for a set of parameters close to quasi-geostrophy must match those of Vallis (See section 9.10, Chapter 9).

### `quick_explokx`
- **Function**: `most_unstable_kx`
- **Purpose**: Finds the most unstable zonal mode (kx) for the given set of parameters.

### `betaExplo`
- **Description**: Finds and plots the maximum growth rates for the vertical modes of the Eady problem as a function of the beta parameter.



