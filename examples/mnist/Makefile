NVCC= nvcc
NVCCFLAGS= -lcublas -g -G

mnist: mnist.cu ../../src/deepcore.cu
	$(NVCC) $(NVCCFLAGS) ./mnist.cu -o ./mnist.exe
	rm -f ./mnist.exp ./mnist.lib ./mnist.pdb ./vc140.pdb
clean:
	rm -f *.o *.exe *.exp *.lib .*pdb
