#ifndef PGM_H
#define PGM_H

typedef int** pgm;

int** load_pgm(const char*, int*, int*);
int** copy_pgm(int**, int, int);
int** null_pgm(int, int);
int   save_pgm(int**, int, int, const char* path);
void  free_pgm(int**, int, int);
int   replace_pgm(int**, int**, int, int);

#endif /* PGM_H */
