CC= nvcc
CFLAGS= -g -G -O3 -lcublas

main: src/main.cu
	$(CC) $(CFLAGS) src/main.cu -o src/main.exe
	src/main.exe

maindebug: src/main.cu
	$(CC) $(CFLAGS) src/main.cu -o src/main.exe
	compute-sanitizer --tool memcheck src/main.exe
	compute-sanitizer --tool racecheck src/main.exe
	compute-sanitizer --tool initcheck src/main.exe
	compute-sanitizer --tool synccheck src/main.exe

test: test/test.cu
	$(CC) $(CFLAGS) test/test.cu -o test/test.exe
	test/test.exe

clean:
	rm -f **/*.o **/*.exe **/*.exp **/*.lib **/*.pdb *.pdb
