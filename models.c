#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#include <stdio.h>  // printf
#include <math.h>
#include "models.h"

void non_convex(IplImage* f_t, IplImage* un_t, int N)
{
    IplImage* u_t = cvCreateImage(cvSize(f_t->width, f_t->height), f_t->depth, f_t->nChannels);
    
    BwImage u(u_t);
    BwImage f(f_t);
    BwImage un(un_t);
    
    int i,
        j,
        x,
        y,
        k,
        c = f_t->nChannels,
        step = f_t->widthStep/sizeof(uchar);

    double ux,
           uy,
           star;

    /*
     * a[0] = a_{i+0.5}
     * a[1] = a_{i-0.5}
     * a[2] = a_{j+0.5}
     * a[3] = a_{j-0.5}
     */
    double a[4],
           d[4],

           epsilon = 0.05,
           lambda  = 0.6,
           t       = 0.1,
           p       = 0.1,

           exponent = ((2-p)/2.0);

    epsilon = pow(epsilon, 2);
    uchar* data = (uchar*)un_t->imageData;

    // TODO: Handle Neumann Boundary Conditions
    for(k = 0; k < N; ++k)
    {
        for(y = 1; y < f_t->width-1; ++y)
        {
            for(x = 1; x < f_t->height-1; ++x)
            {
                // a_{i+.5}
                ux = un[x+1][y]   - un[x][y];
                uy = un[x+1][y+1] + un[x][y+1] -
                     un[x+1][y-1] - un[x][y-1];
                ux = pow(ux, 2);
                uy = pow(uy / 4.0, 2);
                d[0] = pow(epsilon + ux + uy, exponent);

                // a_{i-.5}
                ux = un[x-1][y]   - un[x][y];
                uy = un[x-1][y+1] + un[x][y+1] -
                     un[x-1][y-1] - un[x][y-1];
                ux = pow(ux, 2);
                uy = pow(uy / 4.0, 2);
                d[1] = pow(epsilon + ux + uy, exponent);

                // a_{j+.5}
                uy = un[x][y+1]   - un[x][y];
                ux = un[x+1][y+1] + un[x+1][y] -
                     un[x-1][y+1] - un[x-1][y];
                uy = pow(uy, 2);
                ux = pow(ux / 4.0, 2);
                d[2] = pow(epsilon + ux + uy, exponent);

                // a_{j-.5}
                uy = un[x][y-1]   - un[x][y];
                ux = un[x+1][y-1] + un[x+1][y] -
                     un[x-1][y-1] - un[x-1][y];
                uy = pow(uy, 2);
                ux = pow(ux / 4.0, 2);
                d[3] = pow(epsilon + ux + uy, exponent);

                a[0] = 2.0*d[1]/(d[0] + d[1]);
                a[1] = 2.0*d[0]/(d[0] + d[1]);
                a[2] = 2.0*d[3]/(d[2] + d[3]);
                a[3] = 2.0*d[2]/(d[2] + d[3]);

                star = (
                        a[0]*un[x+1][y]  +
                        a[1]*un[x-1][y]  +
                        a[2]*un[x][y+1]  +
                        a[3]*un[x][y-1]
                       )
                       -
                       ((a[0] + a[1] + a[2] + a[3]) *
                        un[x][y]
                       ) +
                       lambda *
                       (f[x][y] - un[x][y]);
                u[x][y] = un[x][y] + t * star;
            }
        }
        cvCopy(u_t, un_t, NULL);
    }
}
