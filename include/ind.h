/* Definitions for indexing Numpy arrays:                                   */

/* 1D double ndarray:                                                       */
#define INDd(a,i) *((double *)(a->data + i*a->strides[0]))
/* 1D integer ndarray:                                                      */
#define INDi(a,i) *((int *)(a->data + i*a->strides[0]))

