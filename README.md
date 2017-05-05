# pytips
C implementation of the Total Internal Partition Sums (TIPS) code with wrapper for Python

### Table of Contents:
* [Team Members](#team-members)
* [Install and Compile](#install-and-compile)
* [Getting Started](#getting-started)
* [Important Notes](#important-notes)
* [Be Kind](#be-kind)
* [References](#references)
* [License](#license)
* [HITRAN Species List](#hitran-species-list)

### Team Members:
* [Patricio Cubillos](https://github.com/pcubillos/) (UCF) <pcubillos@fulbrightmail.org>
* [AJ Foster](http://aj-foster.com) (UCF)

### Install and Compile:
To obtain the ``pytips`` code, clone the repository to your local machine with the following terminal commands.
```shell
topdir=`pwd`
git clone https://github.com/pcubillos/pytips/
```

Compile the ``pytips`` package:
```shell
cd $topdir/pytips
make  
```

To remove the program binaries, execute:
```shell
make clean
```

### Getting Started:

The following script quickly lets you calculate the partition function for a Methane isotope from the Python Interpreter.  To begin, follow the instructions in the previous Section to install and compile the code.  Now open the Interpreter:

```shell
cd $topdir/pytips
ipython
```

From the Python Interpreter:
```python
import pytips as t

# Display the help message:
help(t)

# HITRAN-2012 ID for Methane (See the full list at the bottom of the page):
molID = 6
# Get the list of isotope ID's for methane:
isoID = t.iso(molID)
print(isoID)
>>> [211 311 212 312]

# Calculate the partition function for all methane isotopes at 100.0 K:
temp1 = 100.0
pf_100K = t.tips(molID, isoID, temp1)
print(pf_100K)
>>> [  116.40320395   232.81455212   939.16979858  1879.84831201]

# Calculate the partition function for the first methane isotope for a range of temperatures:
iso_211 = 211
# Temperatures:
temp2 = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
# Calculate the partition function:
pf_211 = t.tips(molID, iso_211, temp2)
print(pf_211)
>>> [  116.40320395   326.64080103   602.85201367   954.65522363  1417.76400684]

```
### Important Notes:

- TIPS-2011 and HITRAN-2012 agree up to the 42nd (included), the 44th, and
  the 46th species.
- The AFGL code for the 43rd species (C4H2) differ between HITRAN-2012 (2211)
  and TIPS (1221).
- From the 45th species and on, the species ID disagree, being:

  ID  | HITRAN-2012  | TIPS-2011
  ----| -------------| ----
  45  | H2           | C2N2
  46  | CS           | CS
  47  | SO3          | H2
  48  |              | SO
  49  |              | C3H4
  50  |              | CH3
  51  |              | CS2

- To resolve this issue, ``pytips`` will follow the HITRAN-2012 numbering system, and append the remaining species (C2N2, SO, C3H4, CH3, CS2) after the 47th HITRAN species.

### Be Kind:

Please, be kind and aknowledge the effort of the authors, by citing the articles asociated to this project:

  [Cubillos et al. 2015: The Bayesian Atmospheric Radiative-Transifer Code for Exoplanet Modeling](), in preparation.  

### References:

This Python/C code is based on the FORTRAN implementation of the TIPS code
written by R. R. Gamache (Robert_Gamache@uml.edu): [faculty.uml.edu/robert_gamache/software/index.htm](http://faculty.uml.edu/robert_gamache/software/index.htm#TIPS_2011)  
with corresponding publication:   
[Laraia et al. 2011: Total internal partition sums to support planetary remote sensing](http://adsabs.harvard.edu/abs/2011Icar..215..391L)  

### License:

Copyright (c) 2015-2017 Patricio Cubillos and contributors.
pytips is open-source software under the MIT license (see LICENSE).

### HITRAN Species List:

| Molecule Name | Molecule ID   | Isotope ID    |
| ------------- | --------------| --------------|
| H2O           | 1             | 161, 181, 171, 162, 182, 172   |
| CO2      |  2 |    626, 636, 628, 627, 638, 637, 828, 728, 727, 838, 837 |
| O3       |  3 |    666, 668, 686, 667, 676, 886, 868, 678, 768, 786, 776, 767, 888, 887, 878, 778, 787, 777 |
| N2O      |  4 |    446, 456, 546, 448, 447                     |
| CO       |  5 |     26,  36,  28,  27,  38,  37                |
| CH4      |  6 |    211, 311, 212, 312                          |
| O2       |  7 |     66,  68,  67                               |
| NO       |  8 |     46,  56,  48                               |
| SO2      |  9 |    626, 646                                    |
| NO2      | 10 |    646                                         |
| NH3      | 11 |   4111, 5111                                   |
| HNO3     | 12 |    146                                         |
| OH       | 13 |     61,   81,  62                              |
| HF       | 14 |     19                                         |
| HCl      | 15 |     15,   17                                   |
| HBr      | 16 |     19,   11                                   |
| HI       | 17 |     17                                         |
| ClO      | 18 |     56,   76                                   |
| OCS      | 19 |    622,  624,  632,  623,  822                 |
| H2CO     | 20 |    126,  136,  128                             |
| HOCl     | 21 |    165,  167                                   |
| N2       | 22 |     44                                         |
| HCN      | 23 |    124,  134,  125                             |
| CH3Cl    | 24 |    215,  217                                   |
| H2O2     | 25 |   1661                                         |
| C2H2     | 26 |   1221, 1231, 1222                             |
| C2H6     | 27 |   1221, 1231                                   |
| PH3      | 28 |   1111                                         |
| COF2     | 29 |    269                                         |
| SF6      | 30 |     29                                         |
| H2S      | 31 |    121,  141,  131                             |
| HCOOH    | 32 |    126                                         |
| HO2      | 33 |    166                                         |
| O        | 34 |      6                                         |
| ClONO2   | 35 |   5646, 7646                                   |
| NO+      | 36 |     46                                         |
| HOBr     | 37 |    169,  161                                   |
| C2H4     | 38 |    221,  231                                   |
| CH3OH    | 39 |   2161                                         |
| CH3Br    | 40 |    219,  211                                   |
| CH3CN    | 41 |   2124, 2134, 3124, 3134                       |
| CF4      | 42 |     29                                         |
| C4H2     | 43    |  2211(*)                                    |
| HC3N     | 44    |  12224, 12234, 12324, 13224, 12225, 22224   |
| H2       | 45(*) |     11,   12                                | 
| CS       | 46    |     22,   24,     32,    23                 |
| SO3      | 47(*) |     26                                      | 
| C2N2     | 48(*) |   4224, 5225                                | 
| SO       | 49(*) |     26,   46,     28                        | 
| C3H4     | 50(*) |   1221                                      | 
| CH3      | 51(*) |   2111                                      | 
| CS2      | 52(*) |    222,   224,   223,   232                 | 

 (*) See [Notes](#important-notes).
