NVCC= nvcc
NVCCFLAGS= -lcublas

main: src/main.cu
	$(NVCC) $(NVCCFLAGS) src/main.cu -o src/main.exe
	src/main.exe

deepcore: src/deepcore.cu
	$(NVCC) $(NVCCFLAGS) src/deepcore.cu -o src/deepcore.exe
	src/deepcore.exe

test: test/test.cu
	$(NVCC) $(NVCCFLAGS) test/test.cu -o test/test.exe
	test/test.exe

clean:
	rm -f **/*.o **/*.exe **/*.exp **/*.lib **/*.pdb *.pdb
