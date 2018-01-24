// Copyright (c) 2015-2018 Patricio Cubillos and contributors.
// pytips is open-source software under the MIT license (see LICENSE).

/* Lagrange interpolation routines:                                         */

double lagrange3(double *temp, float *q, double t){
  /* LaGrange three-point interpolation:                                    */
  double A0D1, A0D2, A1D1, A1D2, A2D1, A2D2;

  A0D1 = temp[0] - temp[1];
  A0D2 = temp[0] - temp[2];
  A1D1 = temp[1] - temp[0];
  A1D2 = temp[1] - temp[2];
  A2D1 = temp[2] - temp[0];
  A2D2 = temp[2] - temp[1];

  return q[0] * (t-temp[1]) * (t-temp[2]) / (A0D1*A0D2) +
         q[1] * (t-temp[0]) * (t-temp[2]) / (A1D1*A1D2) +
         q[2] * (t-temp[0]) * (t-temp[1]) / (A2D1*A2D2);

}

double lagrange4(double *temp, float *q, double t){
  /* LaGrange four-point interpolation:                                     */
  double A0D1, A0D2, A0D3, A1D1, A1D2, A1D3,
         A2D1, A2D2, A2D3, A3D1, A3D2, A3D3;

  A0D1 = temp[0] - temp[1];
  A0D2 = temp[0] - temp[2];
  A0D3 = temp[0] - temp[3];

  A1D1 = temp[1] - temp[0];
  A1D2 = temp[1] - temp[2];
  A1D3 = temp[1] - temp[3];

  A2D1 = temp[2] - temp[0];
  A2D2 = temp[2] - temp[1];
  A2D3 = temp[2] - temp[3];

  A3D1 = temp[3] - temp[0];
  A3D2 = temp[3] - temp[1];
  A3D3 = temp[3] - temp[2];

  return q[0] * (t-temp[1]) * (t-temp[2]) * (t-temp[3]) / (A0D1*A0D2*A0D3) +
         q[1] * (t-temp[0]) * (t-temp[2]) * (t-temp[3]) / (A1D1*A1D2*A1D3) +
         q[2] * (t-temp[0]) * (t-temp[1]) * (t-temp[3]) / (A2D1*A2D2*A2D3) +
         q[3] * (t-temp[0]) * (t-temp[1]) * (t-temp[2]) / (A3D1*A3D2*A3D3);
}

