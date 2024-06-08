NVCC= nvcc
CFLAGS= -lcublas

train: src/train.cu
	$(NVCC) $(CFLAGS) src/train.cu -o src/train.exe
	src/train.exe

eval: src/eval.cu
	$(NVCC) $(CFLAGS) src/eval.cu -o src/eval.exe
	src/eval.exe

traindebug: src/train.cu
	$(NVCC) $(CFLAGS) src/train.cu -o src/train.exe
	compute-sanitizer --tool memcheck src/train.exe
	compute-sanitizer --tool racecheck src/train.exe
	compute-sanitizer --tool initcheck src/train.exe
	compute-sanitizer --tool synccheck src/train.exe

test: test/test.cu
	$(NVCC) $(CFLAGS) test/test.cu -o test/test.exe
	test/test.exe

clean:
	rm -f **/*.o **/*.exe **/*.exp **/*.lib **/*.pdb *.pdb
