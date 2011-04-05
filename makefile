main : 
	g++ -ggdb `pkg-config --cflags --libs opencv` -o main *.c -O3 -lm;

clean :
	rm *.o main
