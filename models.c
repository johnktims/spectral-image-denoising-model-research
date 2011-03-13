#include <math.h>
#include "pgm.h"

int** laplacian(int** f, int** un, int cols, int rows)
{
    int** u = (int**)copy_pgm(un, cols, rows);
    int x, y;
    for(x = 1; x < cols-1; ++x)
    {
        for(y = 1; y < rows-1; ++y)
        {
            u[x][y] = f[x][y] +
                      0.1 * (
                          f[x + 1][y] +
                          f[x - 1][y] +
                          f[x][y + 1] +
                          f[x][y - 1] -
                          4 * f[x][y]
                      );
        }
    }
    return u;
}

int** tv(int** f, int** un, int cols, int rows)
{
    int** u = (int**)copy_pgm(un, cols, rows);
    int x, y;
    double ux, uy, star;

    /*
     * a[0] = a_{i+0.5}
     * a[1] = a_{i-0.5}
     * a[2] = a_{j+0.5}
     * a[3] = a_{j-0.5}
     */
    double a[4],
           d[4];

    double epsilon = 0.05,
           lambda  = 0.5,
           t       = 1,
           p       = 0.1;

    epsilon = pow(epsilon, 2);

    for(x = 1; x < cols-1; ++x)
    {
        for(y = 1; y < rows-1; ++y)
        {
            // a_{i+.5}
            ux = un[x+1][y]   - un[x][y];
            uy = un[x+1][y+1] + un[x][y+1] -
                 un[x+1][y-1] - un[x][y-1];
            ux = pow(ux, 2);
            uy = pow(uy / 4.0, 2);
            d[0] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{i-.5}
            ux = un[x-1][y]   - un[x][y];
            uy = un[x-1][y+1] + un[x][y+1] -
                 un[x-1][y-1] - un[x][y-1];
            ux = pow(ux, 2);
            uy = pow(uy / 4.0, 2);
            d[1] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{j+.5}
            uy = un[x][y+1]   - un[x][y];
            ux = un[x+1][y+1] + un[x+1][y] -
                 un[x-1][y+1] - un[x-1][y];
            uy = pow(uy, 2);
            ux = pow(ux / 4.0, 2);
            d[2] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{j-.5}
            uy = un[x][y-1]   - un[x][y];
            ux = un[x+1][y-1] + un[x+1][y] -
                 un[x-1][y-1] - un[x-1][y];
            uy = pow(uy, 2);
            ux = pow(ux / 4.0, 2);
            d[3] = pow(epsilon + ux + uy, ((2-p)/2.0));

            a[0] = 2*d[1]/(d[0] + d[1]);
            a[1] = 2*d[0]/(d[0] + d[1]);
            a[2] = 2*d[3]/(d[2] + d[3]);
            a[3] = 2*d[2]/(d[2] + d[3]);

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
    return u;
}

/**
 * Not implemented correctly. Remove (2p-p)/2
 */
int** tvd(int **f, int **un, int cols, int rows)
{
    int** u = (int**)copy_pgm(un, cols, rows);
    int x, y;
    double ux, uy, star;

    /*
     * a[0] = a_{i+0.5}
     * a[1] = a_{i-0.5}
     * a[2] = a_{j+0.5}
     * a[3] = a_{j-0.5}
     */
    double a[4],
           d[4];

    double epsilon = 0.05,
           lambda  = 0.5,
           t       = 1,
           p       = 0.1;

    epsilon = pow(epsilon, 2);

    for(x = 1; x < cols-1; ++x)
    {
        for(y = 1; y < rows-1; ++y)
        {
            // a_{i+.5}
            ux = un[x+1][y]   - un[x][y];
            uy = un[x+1][y+1] + un[x][y+1] -
                 un[x+1][y-1] - un[x][y-1];
            ux = pow(ux, 2);
            uy = pow(uy / 4.0, 2);
            d[0] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{i-.5}
            ux = un[x-1][y]   - un[x][y];
            uy = un[x-1][y+1] + un[x][y+1] -
                 un[x-1][y-1] - un[x][y-1];
            ux = pow(ux, 2);
            uy = pow(uy / 4.0, 2);
            d[1] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{j+.5}
            uy = un[x][y+1]   - un[x][y];
            ux = un[x+1][y+1] + un[x+1][y] -
                 un[x-1][y+1] - un[x-1][y];
            uy = pow(uy, 2);
            ux = pow(ux / 4.0, 2);
            d[2] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{j-.5}
            uy = un[x][y-1]   - un[x][y];
            ux = un[x+1][y-1] + un[x+1][y] -
                 un[x-1][y-1] - un[x-1][y];
            uy = pow(uy, 2);
            ux = pow(ux / 4.0, 2);
            d[3] = pow(epsilon + ux + uy, ((2-p)/2.0));

            a[0] = 2*d[1]/(d[0] + d[1]);
            a[1] = 2*d[0]/(d[0] + d[1]);
            a[2] = 2*d[3]/(d[2] + d[3]);
            a[3] = 2*d[2]/(d[2] + d[3]);

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
    return u;
}

int** non_convex(int** f, int** un, int cols, int rows)
{
    int** u = (int**)copy_pgm(un, cols, rows);
    int x, y;
    double ux, uy, star;

    /*
     * a[0] = a_{i+0.5}
     * a[1] = a_{i-0.5}
     * a[2] = a_{j+0.5}
     * a[3] = a_{j-0.5}
     */
    double a[4],
           d[4];

    double epsilon = 0.05,
           lambda  = 0.5,
           t       = 1,
           p       = 0.1;

    epsilon = pow(epsilon, 2);

    for(x = 1; x < cols-1; ++x)
    {
        for(y = 1; y < rows-1; ++y)
        {
            // a_{i+.5}
            ux = un[x+1][y]   - un[x][y];
            uy = un[x+1][y+1] + un[x][y+1] -
                 un[x+1][y-1] - un[x][y-1];
            ux = pow(ux, 2);
            uy = pow(uy / 4.0, 2);
            d[0] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{i-.5}
            ux = un[x-1][y]   - un[x][y];
            uy = un[x-1][y+1] + un[x][y+1] -
                 un[x-1][y-1] - un[x][y-1];
            ux = pow(ux, 2);
            uy = pow(uy / 4.0, 2);
            d[1] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{j+.5}
            uy = un[x][y+1]   - un[x][y];
            ux = un[x+1][y+1] + un[x+1][y] -
                 un[x-1][y+1] - un[x-1][y];
            uy = pow(uy, 2);
            ux = pow(ux / 4.0, 2);
            d[2] = pow(epsilon + ux + uy, ((2-p)/2.0));

            // a_{j-.5}
            uy = un[x][y-1]   - un[x][y];
            ux = un[x+1][y-1] + un[x+1][y] -
                 un[x-1][y-1] - un[x-1][y];
            uy = pow(uy, 2);
            ux = pow(ux / 4.0, 2);
            d[3] = pow(epsilon + ux + uy, ((2-p)/2.0));

            a[0] = 2*d[1]/(d[0] + d[1]);
            a[1] = 2*d[0]/(d[0] + d[1]);
            a[2] = 2*d[3]/(d[2] + d[3]);
            a[3] = 2*d[2]/(d[2] + d[3]);

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
    return u;
}
