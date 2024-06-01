CC= nvcc
CFLAGS= -O3 -lcublas

main: src/main.cu
	$(CC) $(CFLAGS) src/main.cu -o src/main.exe
	src/main.exe

test: test/test.cu
	$(CC) $(CFLAGS) test/test.cu -o test/test.exe
	test/test.exe

clean:
	rm -f **/*.o **/*.exe **/*.exp **/*.lib
