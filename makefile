main : 
	g++ -ggdb -Wall `pkg-config --cflags --libs opencv` -o main *.c -O3 -lm;

clean :
	rm main
