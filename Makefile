nbod: main.o
	nvcc -O1 -o nbod main.o

main.o: main.cu
	nvcc -O1 -c main.cu

clean:
	rm -f *.o nbod
