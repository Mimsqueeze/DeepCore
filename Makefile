NVCC= nvcc
NVCCFLAGS= -lcublas

main: src/train.cu
	$(NVCC) $(NVCCFLAGS) src/train.cu -o src/train.exe
	src/train.exe

test: test/test.cu
	$(NVCC) $(NVCCFLAGS) test/test.cu -o test/test.exe
	test/test.exe

clean:
	rm -f **/*.o **/*.exe **/*.exp **/*.lib **/*.pdb *.pdb
