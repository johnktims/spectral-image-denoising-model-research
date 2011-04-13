main : 
	g++ -Wall `pkg-config --cflags --libs opencv` -o main main.c models.c -O3 -lm;

nlm-conv : 
	g++ -Wall `pkg-config --cflags --libs opencv` -o nlm-conv nlm-conv.cpp -O3 -lm;

nlm-naive : 
	g++ -Wall `pkg-config --cflags --libs opencv` -o nlm-naive nlm-naive.cpp -O3 -lm;

clean :
	rm main nlm-conv nlm-naive
