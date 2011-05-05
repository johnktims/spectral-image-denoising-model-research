main : 
	g++ -Wall `pkg-config --cflags --libs opencv` -o main main.cpp models.cpp -O3 -lm;

clean :
	rm main
