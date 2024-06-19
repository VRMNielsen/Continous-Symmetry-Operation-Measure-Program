# Continous Symmetry Operation Measure
A program to evaluate the point group symmetry of a coordinating structure.
The program is puplished in association with the following paper in Journal of Physical Chemistry A.
Title:
Authors: 
DOI:

# 2. Description
The program is intended as a tool in coordination chemistry to evaluate the point group symmetry of a coordination complex.
The coordination complex is given as file containing the coordinates and labels for all atoms in the coordination complex. 
The first atom in the series is used to center the complex such that it lies in origo.
Then a set of point groups is manually chosen for evaluation on the complex. 

The evaluation is performed using the Continous Symmetry Operation Measure
CSoM quantifies how well a molecular structure Q is left unchanged by a specific symmetry operation ÔS or how well it is described by a specific point group G. The structure Q is considered to have a point group G if all symmetry operations within a point group can be used on the structure to generate the same structure. The structure Q is considered to be distorted from a point group G if the symmetry operations produce a similar but distorted structure Q’≠Q instead of a structure identical to the original structure ÔSQ = Q. The distortion between a structure and the symmetry operated structure with respect to a specific symmetry operation is called σsym and is calculated with the SOM method as described in Eq. 1.

$ σ_O (Q,Ô_S Q)=∑_(k=1)^N▒〖(|〖Q_k-Ô_S Q_k |〗^2)/(|〖Q_k-Q_0 |〗^2 )×〗 100/N 	Equation. 1.$

The quantification of how well a molecular structure Q belongs to a point group G and, therefore, how well the point group can be used to describe the structure, defined as σsym(G, Q) that is the average distortion between the structure and the structure after all individual symmetry operations within the point group G. How σsym is calculated is described in Eq. 2. Using this formalism, all molecular structures can be evaluated and quantified as σsym-values for all point groups. All operations within a point group are applied to the coordinates of the molecular structure. The σO(Q,ÔSQ) is calculated for the original structure Q which is evaluated against each structure created by a symmetry operation ÔSQ. This is defined in Eq. 3. When Q is operated with a symmetry operation, all coordinates and atom labels need to be compared to the original coordinates and atom labels. These two sets of labels are matched so that the total distance between the two sets of coordinates is minimal. This is done with the Hungarian algorithm75, a solution to the permutation problem, implemented in Python with the Scipy module. The σsym-value is the sum of the σO(Q,ÔSQ)-values normalised by the number of symmetry operations in the point group G.

$ σ_sym (Q,G)=∑_(s=1)^N▒(σ_O (Q,Ô_S Q))/N Equation. 2.$


The program evaluates the point group symmetry of a coordinating complex 

# 3. How to install

# 4. How to Use

# 5. Credits and how to cite




