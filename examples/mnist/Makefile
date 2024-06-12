NVCC= nvcc
NVCCFLAGS= -lcublas

mnist: mnist.cu ../../src/deepcore.cu
	$(NVCC) $(NVCCFLAGS) ./mnist.cu -o ./mnist.exe
	rm -f ./mnist.exp ./mnist.lib
	./mnist.exe

clean:
	rm -f *.o *.exe *.exp *.lib .*pdb
