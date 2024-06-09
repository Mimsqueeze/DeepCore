NVCC= nvcc
NVCCFLAGS= -lcublas

train: ./train.cu
	$(NVCC) $(NVCCFLAGS) ./train.cu -o ./train.exe
	./train.exe

eval: ./eval.cu
	$(NVCC) $(NVCCFLAGS) ./eval.cu -o ./eval.exe
	./eval.exe

clean:
	rm -f *.o *.exe *.exp *.lib .*pdb
