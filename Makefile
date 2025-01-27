program_name := Power
COMPILATOR = nvcc
cuda_compile_flags :=  -g  -gencode arch=compute_61,code=\"sm_61,compute_61\"   -O3   -shared -Xcompiler  -fPIC -Wno-deprecated-gpu-targets -std=c++11

$(program_name) : main.o  
	$(COMPILATOR) main.o    -o ./$@

main.o : main.cu LibIntGPU.cu cuFuncs.cu
	$(COMPILATOR) $(cuda_compile_flags)  -c  main.cu

clean:
	rm -f *.o *.so binary
