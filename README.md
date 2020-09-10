# profiles-classification

Classification of profiles from:

1. [x]Argo (Rosso et al., 2020)
2. [] seal and hydrographic data 
3. [] and model output, 

using a **Profile Classification Model** (PCM; Maze et al., 2017).

Few useful libraries that I use in the scripts:

    - [Gibbs-SeaWater Oceanographic Toolbox](http://www.teos-10.org/software.htm) contains the TEOS-10 subroutines for evaluating the thermodynamic properties of pure water

    - [cmocean](https://matplotlib.org/cmocean/) for nice colormaps. Cmocean is found also in [Palettable](https://jiffyclub.github.io/palettable/), which contains different and very cool colormaps (I really like [colorbrewer](http://colorbrewer2.org)). For installing the libraries using Anaconda: e.g. Palettable: conda install -c conda-forge palettable
    
    - [scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms
    
    - numpy, scipy and matplotlib.
    

### References
Maze, G., Mercier, H., & Cabanes, C. (2017). Profile Classification Models. Mercator Ocean Journal, 55, 48–56.

Rosso, I., Mazloff, M. R., Talley, L. D., Purkey, S. G., Freeman, N. M., & Maze, G. (2020). Water mass and biogeochemical variability in the Kerguelen sector of the Southern Ocean: A machine learning approach for a mixing hot spot. Journal of Geophysical Research: Oceans, 125, e2019JC015877. https://doi.org/10.1029/2019JC015877
