/**
 * compile : gcc main.c -O3 -lm
 * make gif: convert -delay 10 -loop 0 Iteration_*pgm out.gif
 */

#include <stdio.h>  // printf
#include "pgm.h"
#include "models.h"

int main(char argc, char *argv[])
{
    int cols, rows, x, y, i, I = 100;

    // Load noisy pgm `f` with Neumann B.C.s
    int** f;
    
    // Let `u` be the original pgm
    int** u;
    
    // load the passed in pgm or try to open
    // the lena pgm in this directory
    if(argc > 1)
    {
        f = load_pgm(argv[1], &cols, &rows);
    }
    else
    {
        f = load_pgm("lena_f.pgm", &cols, &rows);
        u = (int**)copy_pgm(f, cols, rows);
    }

    if(!f)
    {
        puts("Invalid PGM file.");
        puts("Syntax: ./program file.pgm");
        return 0;
    }
    printf("Image dimensions: %d by %d\n", cols, rows);

    // Let `ui` be our current interation
    int** ui = (int**)copy_pgm(u, cols, rows);

    // The filename of the current iteration
    char buffer[19];

    // Apply PDE to pgm `I` times
    for(i = 0; i < I; ++i)
    {
        sprintf(buffer, "Iteration_%04d.pgm", i);
        ui = non_convex(f, u, cols, rows);
        save_pgm(ui, cols, rows, buffer);
        replace_pgm(u, ui, cols, rows);
    }

    free_pgm(f, cols, rows);
    free_pgm(u, cols, rows);
    free_pgm(ui, cols, rows);
    return 0;
}

