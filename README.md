# taslib

This project seeks to translate the main parts of the TasLib used by Six from C to Python. TasLib is the hind-lying mathematical codes use to calculate reciprocal lattice, scattering angles, UB matrices, and all other sort of math connected to the scattering geometry in a neutron scattering experiment.

## Change of convention
As the TasLib code is from the previous millennium, it is of no surprise that the conventions regarding scattering has changed slightly. In particular, the definition of the reciprocal lattice vectors has changed by a scaling of two pi. This change shows up many places throughout the code, including the definitions of the B matrix and in the calculations of scattering vector lengths. In this python version the convention has been updated, and there is no need of a factor 2 pi every now and again.



## Assumption
By default, it is assumed that taslib is correct until proven guilty. This results in some of the tests being simply a assertion that the python generated values match those from the c version. 
### Error in code
As a matter of fact, only one place has an error been found, in the *cellFromUB* function in the *cell.c* file. Here the _direct->gamma = = Acosd(G[0][1] / (direct->a * direct->c));_ should become _direct->gamma = = Acosd(G[0][1] / (direct->a * direct->b));_. That is, direct->c should be replaced by direct->b.
