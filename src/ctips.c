#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>

#include "ind.h"
#include "ctips.h"
#include "lagrange.h"


int binsearch(double *array, double value, int lo, int hi){
  /* Binary search of value in a sorted array, between indices lo, and hi
     (with lo < hi).
     Returns the index of the point inmediately smaller or equal to value.  */
  if (hi-lo <= 1)
    return lo;
  if (array[(hi+lo)/2] > value)
    return binsearch(array, value, lo, (hi+lo)/2);
  return binsearch(array, value, (hi+lo)/2, hi);
}


int qindex(int molID, int iso){
  /* Get the Qdata (starting) index for the given isotope ID of
     molecule molID.                                                        */
  int i;
  for (i=0; i<niso[molID]; i++)
    if (isoID[cumiso[molID]+i] == iso)
      return ntemp*(cumiso[molID]+i);
  return -1;
}


static double Qeval(double temperature, int itemp, float *Qdat){
  /* Evaluate the Lagrange interpolation of array Qdat at the given
     temperature (which is in index itemp)                                  */
  double Q;

  if (itemp == 0)
    Q = lagrange3(Tdat+itemp,   Qdat+itemp,   temperature);
  else if (itemp == ntemp-2)
    Q = lagrange3(Tdat+itemp-1, Qdat+itemp-1, temperature);
  else
    Q = lagrange4(Tdat+itemp-1, Qdat+itemp-1, temperature);
  return Q;
}


PyDoc_STRVAR(iso__doc__,
"Return the list of isotope IDs for a given molecule ID.  \n\
                                                          \n\
Parameters:                                               \n\
-----------                                               \n\
molID: integer                                            \n\
   Molecule ID (as given in HITRAN 2012).                 \n\
                                                          \n\
Returns:                                                  \n\
--------                                                  \n\
isoID: 1D integer ndarray                                 \n\
   Array of isotope IDs (as given in HITRAN 2012).        \n\
");

static PyObject *iso(PyObject *self, PyObject *args){
  PyArrayObject *iso;  /* Output isotope ID arrays                          */
  int mol,             /* Input molecule ID                                 */
      i;               /* for loop index                                    */
  npy_intp size[1];

  /* Load inputs:                                                           */
  if (!PyArg_ParseTuple(args, "i", &mol))
    return NULL;

  /* Size of output array:                                                  */
  size[0] = niso[mol];

  /* Fill in isotope ID's for this molecule:                                */
  iso = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_INT);
  for (i=0; i<niso[mol]; i++){
    INDi(iso, i) = isoID[cumiso[mol]+i];
  }

  return Py_BuildValue("N", iso);
}
 

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
");

static PyObject *tips(PyObject *self, PyObject *args){

  PyArrayObject *molID, *isoID, *temperature, /* Input arrays               */
                *Qarray; /* Output partition-function array                 */
  double temp=-1.0;      /* Current temperature                             */
  int mol=-1, iso=-1,    /* Current molecule and isotope                    */
      iiso=-1,           /* Q index for given isotope                       */
      itemp=-1,          /* Temperature index from table                    */
      i, ndata;          /* Input arrays length                             */
  npy_intp size[1];

  /* Load inputs:                                                           */
  if (!PyArg_ParseTuple(args, "OOO", &molID, &isoID, &temperature))
    return NULL;

  /* Get array size:                                                        */
  ndata = PyArray_DIM(molID, 0);
  size[0] = ndata;

  /* Returned array with the partition function:                            */
  Qarray = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

  /* Calculate the partition function for each input value:                 */
  for (i=0; i<ndata; i++){
    /* Update the molecule, isotope indices only if necessary:              */
    if (INDi(molID,i) != mol || INDi(isoID,i) != iso){
      /* Get the isotope index in the table:                                */
      iiso = qindex(INDi(molID,i), INDi(isoID,i));
      if (iiso < 0){
        printf("Molecule ID '%d' does not have isotope ID '%d'.\n",
               INDi(molID,i), INDi(isoID,i));
        INDd(Qarray,i) = 0.0;
        continue;
      }
      /* Get the partition function array for this isotope:                 */
      mol = INDi(molID,i);
      iso = INDi(isoID,i);
    }
    /* Get the index of Tdat inmediately <= temp[i]:                        */
    if (INDd(temperature,i) != temp){
      if (INDd(temperature,i) < 70.0 || INDd(temperature,i) > 3000.0){
        printf("Temperature %.1f is out of bounds: 70.0 - 3000.0 K.\n",
               INDd(temperature,i));
        INDd(Qarray,i) = 0.0;
        continue;
      }
      itemp = binsearch(Tdat, INDd(temperature,i), 0, ntemp);
      temp = INDd(temperature,i);
    }
    /* Calculate the Partition function at the requested temperature:       */
    INDd(Qarray,i) = Qeval(temp, itemp, Qdata+iiso);
  }

  return Py_BuildValue("N", Qarray);
}


/* The module doc string                                                    */
PyDoc_STRVAR(ctips__doc__,
"Python wrapper for Total Internal Partition Sum (TIPS) calculation.\n\
                                                                    \n\
Written by:                                                         \n\
-----------                                                         \n\
Patricio Cubillos,  University of Central Florida.                  \n\
                    pcubillos@fulbrightmail.org.                    \n\
                                                                    \n\
Notes:                                                              \n\
------                                                              \n\
This program is based on the FORTRAN implementation of TIPS         \n\
written by R.R. Gamache (Robert_Gamache@uml.edu):                   \n\
  http://faculty.uml.edu/robert_gamache/software/index.htm#TIPS_2011\n\
with corresponding publication: JQSRT - 82, 401-412, 2003           \n\
  J. Fischer R.R. Gamache, A. Goldman, L.S. Rothman, A. Perrin      \n\
");

/* A list of all the methods defined by this module.                        */
static PyMethodDef ctips_methods[] = {
    {"tips",      tips,      METH_VARARGS, tips__doc__},
    {"iso",       iso,       METH_VARARGS, iso__doc__},
    {NULL,         NULL,       0,            NULL}       /* sentinel        */
};


#if PY_MAJOR_VERSION >= 3

/* Module definition for Python 3.                                          */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ctips",
    ctips__doc__,
    -1,
    ctips_methods
};

/* When Python 3 imports a C module named 'X' it loads the module           */
/* then looks for a method named "PyInit_"+X and calls it.                  */
PyObject *PyInit_ctips (void) {
  PyObject *module = PyModule_Create(&moduledef);
  import_array();
  return module;
}

#else // Python 2

/* When Python 2 imports a C module named 'X' it loads the module           */
/* then looks for a method named "init"+X and calls it.                     */
void initctips(void){
  Py_InitModule3("ctips", ctips_methods, ctips__doc__);
  import_array();
}

#endif
