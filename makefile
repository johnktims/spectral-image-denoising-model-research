main : pgm.o models.o
	gcc -o main pgm.o models.o main.c -lm -O3

pgm.o : pgm.c pgm.h
	gcc -c pgm.c

models.o : models.c models.h
	gcc -c models.c

clean :
	rm *.o main
