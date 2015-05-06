#include <Python.h>
#include <numpy/arrayobject.h>
#include "tips.h"

/* Access to i-th value of array a:                                         */
#define INDd(a,i) *((double *)(a->data + i*a->strides[0]))
#define INDi(a,i) *((int    *)(a->data + i*a->strides[0]))

PyDoc_STRVAR(tips__doc__,
"Calculate the partition function for the given (HITRAN) isotopes   \n\
and temperatures (70-3000K) using a 4-point Lagrange interpolation. \n\
                                                                    \n\
Parameters:                                                         \n\
-----------                                                         \n\
molID: 1D integer ndarray                                           \n\
   Array of molecules ID (as given in HITRAN).                      \n\
isoID: 1D integer ndarray                                           \n\
   Array of isotopes ID (as given in HITRAN).                       \n\
temperature: 1D double ndarray                                      \n\
   Array of temperatures to evaluate the partition function.        \n\
                                                                    \n\
Written by:                                                         \n\
-----------                                                         \n\
Patricio Cubillos,  University of Central Florida.                  \n\
                    pcubillos@fulbrightmail.org,   2015-02-10.      \n\
                                                                    \n\
Notes:                                                              \n\
------                                                              \n\
This program is based on the FORTRAN implementation of TIPS         \n\
written by R.R. Gamache (Robert_Gamache@uml.edu):                   \n\
  http://faculty.uml.edu/robert_gamache/software/index.htm#TIPS_2011\n\
with corresponding publication: JQSRT - 82, 401-412, 2003           \n\
  J. Fischer R.R. Gamache, A. Goldman, L.S. Rothman, A. Perrin");

static PyObject *tips(PyObject *self, PyObject *args){

  PyArrayObject *molID, *isoID, *temperature, /* Input arrays               */
                *Qarray; /* Output partition-function array                 */
  double temp=-1.0;      /* Current temperature                             */
  int mol=-1, iso=-1,    /* Current molecule and isotope                    */
      iiso,              /* Isotope sequential index for the given molecule */
      itemp=-1,          /* Temperature index from table                    */
      i, ndata;          /* Input arrays length                             */
  float *Qdat;           /* Tabulated Q(T) array for this isotope           */
  npy_intp size[1];

  /* Load inputs:                                                           */
  if (!PyArg_ParseTuple(args, "OOO", &molID, &isoID, &temperature))
    return NULL;

  /* Get array size:                                                        */
  ndata = molID->dimensions[0];
  size[0] = ndata;

  /* Returned array with the partition function:                            */
  Qarray = (PyArrayObject *) PyArray_SimpleNew(1, size, PyArray_DOUBLE);

  for (i=0; i<ndata; i++){
    /* Update the molecule, isotope indices only if necessary:              */
    if (INDi(molID,i) != mol || INDi(isoID,i) != iso){
      /* Get the isotope index:                                             */
      iiso = getiso(INDi(molID,i), INDi(isoID,i));
      if (iiso < 0){
        printf("Molecule ID '%d' does not have isotope ID '%d'.\n",
               INDi(molID,i), INDi(isoID,i));
        INDd(Qarray,i) = 0.0;
        continue;
      }
      /* Get the partition function array:                                  */
      getq(INDi(molID,i), iiso, &Qdat);
      mol = INDi(molID,i);
      iso = INDi(isoID,i);
    }
    /* Get the index of Tdat inmediately <= temp[i]:                        */
    if (INDd(temperature,i) != temp){
      itemp = binsearch(Tdat, INDd(temperature,i), 0, ntemp);
      temp = INDd(temperature,i);
    }
    /* Calculate the Partition function at the requested temperature:       */
    if (itemp == 0)
      INDd(Qarray,i) = lagrange3(Tdat+itemp,   Qdat+itemp,   temp);
    else if (itemp == ntemp-2)
      INDd(Qarray,i) = lagrange3(Tdat+itemp-1, Qdat+itemp-1, temp);
    else
      INDd(Qarray,i) = lagrange4(Tdat+itemp-1, Qdat+itemp-1, temp);
  }

  //Py_XDECREF(size);
  return Py_BuildValue("N", Qarray);
}

/* The module doc string                                                    */
PyDoc_STRVAR(ctips__doc__,
  "Python wrapper for Total Internal Partition Sum (TIPS) calculation.");

/* A list of all the methods defined by this module.                        */
static PyMethodDef ctips_methods[] = {
    {"tips",      tips,      METH_VARARGS, tips__doc__},
    {NULL,         NULL,       0,            NULL}       /* sentinel        */
};


/* When Python imports a C module named 'X' it loads the module             */
/* then looks for a method named "init"+X and calls it.                     */
void initctips(void){
  Py_InitModule3("ctips", ctips_methods, ctips__doc__);
  import_array();
}
