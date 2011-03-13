#include <stdio.h>  // file functions
#include <stdlib.h> // malloc, free
#include <string.h> // memmove

int** load_pgm(const char* path, int* cols, int* rows)
{
    FILE *fp = fopen(path, "rb");

    // Check for valid file path
    if(!fp)
    {
        return NULL;
    }

    /*
     * Skip the first two rows because:
     *   - Line 1 is the type of netpbm format(Ex: P2 for grayscale)
     *   - Line 2 is a one-line comment
     */
    int i = 0,
        j = 0;
    while(i < 2)
    {
        if(fgetc(fp) == '\n')
        {
            ++i;
        }
    }

    // Columns in pgm
    fscanf(fp, "%d", cols);

    // Rows in pgm
    fscanf(fp, "%d", rows);

    // Largest value in matrix (Discard)
    fscanf(fp, "%d", &i);

    /*
     * Increase size out of consideration for Neumann B.C.s
     *   - 2 extra columns: 1 on the left and one on the right
     *   - 2 extra rows: 1 on the bottom and one on the top
     */
    *cols += 2;
    *rows += 2;

    int** pgm = (int**)malloc(sizeof(int*)*(*rows));
    for(i = 0; i < *rows; ++i)
    {
        pgm[i] = (int*)malloc(sizeof(int)*(*cols));
    }

    /*
     * Fill array
     *
     * Skip spaces for Neumann B.C.s until we've
     * read in all of the pgm data
     */
    int d;
    for(i = 1; i < *cols-1; ++i)
    {
        for(j = 1; j < *rows-1; ++j)
        {
            if(feof(fp))
            {
                break;
            }

            // Translate x,y coordinates to array offset

            // Read the next number into our pgm array
            fscanf(fp, "%d", &d);
            pgm[i][j] = d;
        }
    }

    /*
     * Handle Neumann B.C.s
     */

    // Left and Right Columns
    for(i = 0; i < *rows; ++i)
    {
        pgm[0][i]       = pgm[2][i];
        pgm[*cols-1][i] = pgm[*cols-3][i];
    }

    // Top and bottom rows
    for(i = 0; i < *cols; ++i)
    {
        pgm[i][0] = pgm[i][2];
        pgm[i][*rows-1] = pgm[*rows-3][i];
    }

    fclose(fp);

    return pgm;
}

int** null_pgm(int cols, int rows)
{
    int** u = (int**)malloc(sizeof(int*)*rows);
    int i;
    for(i = 0; i < rows; ++i)
    {
        u[i] = (int*)malloc(sizeof(int)*(cols));
    }

    return u;
}

int** copy_pgm(int** src, int cols, int rows)
{
    int** u = (int**)null_pgm(cols, rows);
    int i;
    for(i = 0; i < rows; ++i)
    {
        memmove(u[i], src[i], sizeof(int)*cols);
    }

    return u;
}

int save_pgm(int** pgm, int cols, int rows, const char *path)
{
    int i = 0,
        j = 0;
    FILE *fp = fopen(path, "wb");
    if(!fp)
    {
        return 0;
    }

    // Write header
    fprintf(fp, "P2\n#\n%d %d\n255", cols, rows);

    // Write array
    for(i = 0; i < cols; ++i)
    {
        for(j = 0; j < rows; ++j)
        {
            fprintf(fp, "\n%d", pgm[i][j]);
        }
    }

    fclose(fp);
    return 1;
}

void free_pgm(int** src, int cols, int rows)
{
    int i;
    for(i = 0; i < rows; ++i)
    {
        free(src[i]);
    }
    free(src);
}

void replace_pgm(int** dst, int** src, int cols, int rows)
{
    int i;
    for(i = 0; i < rows; ++i)
    {
        memmove(dst[i], src[i], sizeof(int)*cols);
    }
}
