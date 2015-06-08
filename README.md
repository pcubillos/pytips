# ctips
C implementation of the Total Internal Partition Sums (TIPS) code with wrapper for Python

### Table of Contents:
* [Team Members](#team-members)
* [Install and Compile](#install-and-compile)
* [Getting Started](#getting-started)
* [Be Kind](#be-kind)
* [References](#references)
* [License](#license)
* [HITRAN Species List](#hitran-species-list)

### Team Members:
* [Patricio Cubillos](https://github.com/pcubillos/) (UCF) <pcubillos@fulbrightmail.org>

### Install and Compile:
To obtain the CTIPS code, clone the repository to your local machine with the following terminal commands.  First, create a top-level directory to place the code:  
```shell
mkdir ctips_demo/  
cd ctips_demo/  
topdir=`pwd`
```

Clone the repository to your working directory:  
```shell
git clone https://github.com/pcubillos/ctips/
```

Compile ctips program:
```shell
cd $topdir  
make  
```

To remove the program binaries, execute:
```shell
make clean
```

### Getting Started:

The following script quickly lets you calculate the partition function for a Methane isotope from the Python Interpreter.  To begin, follow the instructions in the previous Section to install and compile the code.  Now open the Interpreter:

```shell
cd $topdir
ipython
```

From the Python Interpreter:
```python
import sys
sys.path.append("ctips/lib")
import ctips as ct

# Display the help message:
help(ct.tips)

# HITRAN ID for Methane (See the full list at the bottom of the page):
molID = 6
# Let's use the most common isotope of Methane:
isoID = 211
# Temperatures:
temp = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

# The temp, molID, and isoID arrays must all have the same length:
molID = np.repeat(molID, len(temp))
isoID = np.repeat(isoID, len(temp))

# Calculate the partition function:
pf = ct.tips(molID, isoID, temp)
# Show results:
print(pf)
>>> [  116.40320395   326.64080103   602.85201367   954.65522363  1417.76400684]

```

### Be Kind:

Please, be kind and aknowledge the effort of the authors, by citing the articles asociated to this project:

  [Cubillos et al. 2015: The Bayesian Atmospheric Radiative-Transifer Code for Exoplanet Modeling](), in preparation.  

### References:

This C code is based on the FORTRAN implementation of the TIPS code
written by R. R. Gamache (Robert_Gamache@uml.edu): [faculty.uml.edu/robert_gamache/software/index.htm](http://faculty.uml.edu/robert_gamache/software/index.htm#TIPS_2011)  
with corresponding publication:   
[Laraia et al. 2011: Total internal partition sums to support planetary remote sensing](http://adsabs.harvard.edu/abs/2011Icar..215..391L)  

### License:

We will add a License as soon as the original author of the FORTRAN TIPS code puts a License for his code.

### HITRAN Species List:

| Molecule Name | Molecule ID   | Isotope ID    |
| ------------- | --------------| --------------|
| H2O           | 1             | 161, 181, 171, 162, 182, 172 |
| CO2      |  2 |    626, 636, 628, 627, 638, 637, 828, 728, 727, 838, 837, 666, 668, 686, 667, 676, 886, 868, 678, 768                          |              
| O3       |  3 |    786, 776, 767, 888, 887, 878, 778, 787, 777 |
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
| NO+      | 36 |     46,                                        |
| HOBr     | 37 |    169,  161                                   |
| C2H4     | 38 |    221,  231                                   |
| CH3OH    | 39 |   2161                                         |
| CH3Br    | 40 |    219,  211                                   |
| CH3CN    | 41 |   2124, 2134, 3124, 3134                       |
| CF4      | 42 |     29                                         |
| C4H2     | 43 |   1221                                         |
| HC3N     | 44 |  12224, 12234, 12324, 13224, 12225, 22224      |
| C2N2     | 45 |   4224, 5225                                   |
| CS       | 46 |     22,   24,     32,    23                    |
| H2       | 47 |     11,   12                                   |
| SO       | 48 |     26,   46,     28                           |
| C3H4     | 49 |   1221                                         |
| CH3      | 50 |   2111                                         |
| CS2      | 51 |    222,   224,   223,   232                    |
