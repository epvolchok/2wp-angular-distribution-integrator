
#include <cuda.h>
#include <cuComplex.h>
//#include <python3.6/Python.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#if FLOATTYPE==0
	#define FLOAT double
	#define cuFLOATComplex  cuDoubleComplex
	#define make_cuFLOATComplex  make_cuDoubleComplex
#endif

#if FLOATTYPE==1
	#define FLOAT float
	#define cuFLOATComplex  cuFloatComplex
	#define make_cuFLOATComplex  make_cuFloatComplex
	#define cuCmul cuCmulf
	#define cuCadd cuCaddf
	#define cuCdiv cuCdivf
	#define cuCreal cuCrealf
	#define cuCimag cuCimagf

#endif


struct cuComplex3
{
	cuFLOATComplex x, y, z;
};



int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta
      if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

int assigned_device;
void SetDevice(int gpu)
{
  assigned_device=gpu;

int used_device;
  // Select the used device:
  if ( cudaSetDevice(assigned_device) != cudaSuccess or
      cudaGetDevice( &used_device ) != cudaSuccess or
      used_device != assigned_device) {
    printf ("Error: unable to set device %d\n", assigned_device);
  }
cudaDeviceProp prop;

cudaGetDeviceProperties(&prop,assigned_device);
  int cores;
  cores=getSPcores(prop);
  cout<<"multiProcessorCount = "<<prop.multiProcessorCount<<endl;
  cout<<"maxThreadsPerBlock = "<< prop.maxThreadsPerBlock<<endl;
  cout<<"maxGridSize[0] = "<< prop.maxGridSize[0]<<endl;
  cout<<"l2CacheSize = "<<prop.l2CacheSize<<endl;
  cout<<"regsPerBlock = "<<prop.regsPerBlock<<endl;
  cout<<"memoryClockRate = "<<prop.memoryClockRate<<endl;
  cout<<"sharedMemPerBlock = "<<prop.sharedMemPerBlock<<endl;
  cout<<"totalGlobalMem = "<<prop.totalGlobalMem<<endl;
  cout<<"totalConstMem = "<<prop.totalConstMem<<endl;
  cout<<"clockRate = "<<prop.clockRate<<endl;
  cout<<"name = "<<prop.name<<endl;
  cout<<"cores = "<<cores<<endl;
  cout<<endl;
  
}

__device__ __host__ cuFLOATComplex  operator*(cuFLOATComplex a, cuFLOATComplex b) { return cuCmul(a,b); }
__device__ __host__ cuFLOATComplex  operator+(cuFLOATComplex a, cuFLOATComplex b) { return cuCadd(a,b); }
__device__ __host__ cuFLOATComplex  operator/(cuFLOATComplex a, cuFLOATComplex b) { return cuCdiv(a,b); }
__device__ __host__ cuFLOATComplex  operator-(cuFLOATComplex a, cuFLOATComplex b) { return make_cuFLOATComplex(cuCreal(a)-cuCreal(b),cuCimag(a)-cuCimag(b));}
__device__ __host__ cuFLOATComplex  operator-(cuFLOATComplex a) { return make_cuFLOATComplex(-cuCreal(a),-cuCimag(a));}
//операции с обычными вещественными числами
__device__ __host__ cuFLOATComplex  operator*(cuFLOATComplex a, FLOAT b) { return make_cuFLOATComplex(cuCreal(a)*b,cuCimag(a)*b); }
__device__ __host__ cuFLOATComplex  operator/(cuFLOATComplex a, FLOAT b) { return make_cuFLOATComplex(cuCreal(a)/b,cuCimag(a)/b); }
__device__ __host__ cuFLOATComplex  operator+(cuFLOATComplex a, FLOAT b) { return make_cuFLOATComplex(cuCreal(a)+b,cuCimag(a)); }
__device__ __host__ cuFLOATComplex  operator-(cuFLOATComplex a, FLOAT b) { return make_cuFLOATComplex(cuCreal(a)-b,cuCimag(a)); }


__device__ __host__ cuFLOATComplex  operator*(FLOAT a, cuFLOATComplex b) { return make_cuFLOATComplex(cuCreal(b)*a,cuCimag(b)*a); }
__device__ __host__ cuFLOATComplex  operator/(FLOAT a, cuFLOATComplex b) { return cuCdiv(make_cuFLOATComplex(a,0),b); }
__device__ __host__ cuFLOATComplex  operator+(FLOAT a, cuFLOATComplex b) { return make_cuFLOATComplex(cuCreal(b)+a,cuCimag(b)); }
__device__ __host__ cuFLOATComplex  operator-(FLOAT a, cuFLOATComplex b) { return make_cuFLOATComplex(a-cuCreal(b),-cuCimag(b)); }

__device__ __host__ cuFLOATComplex  exp(cuFLOATComplex a) {

FLOAT x=cuCreal(a);
FLOAT y=cuCimag(a);
// exp(x) * (cos(y) + i sin(y))
	return exp(x)*(cos(y) +  make_cuFLOATComplex(0,1.)*sin(y));

	}
	

	
__constant__ FLOAT co_FuncParam[MAXCONSTPARAM];

#include "cuFuncs.cu"





#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600)
#else
#if FLOATTYPE == 0
__device__ double atomicAdd(double* address, double val) //не передавать в val конструкции типа "threadIdx.x*...", будет бред...
{
	unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif
#endif

extern __shared__ FLOAT shared[];



extern "C" void CopyToConstantMem(int ParamNum,FLOAT *FuncConstParam)
{
	cudaMemcpyToSymbol(co_FuncParam,FuncConstParam,ParamNum*sizeof(FLOAT));

}


__global__ void KernelParallelIntegrator(FLOAT *result, int multiplicity, FLOAT *IntParams, FLOAT *FuncParam, FLOAT *dX, int *TotalN, FLOAT *LeftLim) //ядро
{
	
	int indeX=blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ FLOAT shRes[6];
	
	if(threadIdx.x<6)
		shRes[threadIdx.x]=0;
		__syncthreads();
	
	int count=CompCount;
	FLOAT *locresult;
	locresult=new FLOAT[count];
	for (int i=0; i<count; i++)
	locresult[i]=0;
	
	FLOAT *X;//координата по каждому интегралу
	X=new FLOAT[multiplicity];
	int *ind;
	ind=new int[multiplicity];
	
	cuComplex3 tempRes;
	tempRes.x=make_cuFLOATComplex(0, 0);
	tempRes.y=make_cuFLOATComplex(0, 0);
	tempRes.z=make_cuFLOATComplex(0, 0);
	
	for(int index=indeX;index<TotalN[0];index+=blockDim.x * gridDim.x)
	{
		ind[0]=index;
		X[0]=ind[0]/TotalN[1];
		for(int n=1;n<multiplicity;n++)//для каждого индекса вычислить соответствующий отрезок интегрирования
		{
			ind[n]=ind[n-1]-X[n-1]*TotalN[n];
			X[n]=ind[n]/TotalN[n+1];
		}
		for(int n=0;n<multiplicity;n++)
		{
			X[n]=LeftLim[n]+X[n]*dX[n]+dX[n]*0.5;
		}
		
		Current(X, FuncParam, &tempRes);
		
		locresult[0]+=cuCreal(tempRes.x);
		locresult[1]+=cuCreal(tempRes.y);
		locresult[2]+=cuCreal(tempRes.z);
		
		locresult[3+0]+=cuCimag(tempRes.x);
		locresult[3+1]+=cuCimag(tempRes.y);
		locresult[3+2]+=cuCimag(tempRes.z);
		//printf("Re(J_res.x) %f, Im(J_res.x) %f, Re(J_res.y) %f, Im(J_res.y) %f, Re(J_res.z) %f, Im(J_res.z) %f \n",  locresult[0], locresult[0+3], locresult[1], locresult[1+3], locresult[2], locresult[2+3]);
	}
	
	for(int n=0;n<multiplicity;n++)
		{	for (int i=0; i<count; i++)
			locresult[i]*=dX[n];
			
		}
		
		//if(FuncParam[1]>2.7)
		//printf("theta %f, phi %f, Re(J_res.x) %f, Im(J_res.x) %f, Re(J_res.y) %f, Im(J_res.y) %f, Re(J_res.z) %f, Im(J_res.z) %f \n", FuncParam[1], FuncParam[2],  locresult[0], locresult[0+3], locresult[1], locresult[1+3], locresult[2], locresult[2+3]);
		//printf("%f \n", locresult[0]);
//__syncthreads();//синхронизация нитей
		//printf("%g + I %g \n", locresult[0], locresult[1] );

	for (int i=0; i< 6; i++)
		atomicAdd(&shRes[i],locresult[i]);
		
		__syncthreads();//синхронизация нитей

//	for(int thr=threadIdx.x;thr<count; thr+=blockDim.x)
//		atomicAdd(&result[thr], shRes[thr]);
//	printf("%d,  %g + I %g \n", blockIdx.x,shRes[0], shRes[1] );


	if(threadIdx.x<6)
		{
		atomicAdd(&result[threadIdx.x],shRes[threadIdx.x]);
		}
//	for (int i=0; i< 6; i++)
//		atomicAdd(&result[i],locresult[i]);

delete X;
delete ind;
delete locresult;
	__syncthreads();
	
	//if(threadIdx.x==0)
//printf(" Re(J_res.x) %f, Im(J_res.x) %f, Re(J_res.y) %f, Im(J_res.y) %f, Re(J_res.z) %f, Im(J_res.z) %f \n",  result[0], result[0+3], result[1], result[1+3], result[2], result[2+3]);
}



//extern "C" void CopyToConstantMem(int ParamNum,FLOAT *FuncConstParam)
//{
	//cudaMemcpyToSymbol(co_FuncParam,FuncConstParam,ParamNum*sizeof(FLOAT));

//}

void ParallelNquadIntegrator(int multiplicity, FLOAT *IntParams, int NumParam, FLOAT *FuncParam, FLOAT *J_res)
//J_res[CompCount]
{
	int Nblocks=NBLOCKS;
	int Nthreads=NTHREADS;
	
	int count=CompCount;
	FLOAT *dev_IntParams, *dev_LeftLim;
	FLOAT *dev_FuncParam;
	
	int *dev_TotalN,*cpu_TotalN;
	FLOAT *dev_dX,*cpu_dX, *cpu_LeftLim;
	
	FLOAT *dev_J;

    cudaMalloc(&dev_TotalN, (multiplicity+1)*sizeof(int)); //выделяет (multiplicity+1)*sizeof(int) памяти на девайсе под dev_TotalN (возвращает указатель на участок памяти)
    cudaMalloc(&dev_dX, multiplicity*sizeof(FLOAT));
    cudaMalloc(&dev_LeftLim, multiplicity*sizeof(FLOAT));
    
    cpu_TotalN=new int[multiplicity+1]; //массив из общего числа отрезков N1*N2*N3, N2*N3, N3...
    cpu_dX=new FLOAT[multiplicity];
    cpu_LeftLim=new FLOAT[multiplicity]; //элементарные отрезки по каждому интегралу
    cpu_TotalN[0]=1;

	for(int i=0;i<multiplicity;i++)
		cpu_TotalN[0]*=IntParams[2+i*3]; //N1*N2*N3

	for(int i=0;i<multiplicity;i++)
	{
		cpu_dX[i]=(IntParams[1+i*3]-IntParams[0+i*3])/IntParams[2+i*3]; //(правый-левый)/количество = элементарный отрезок
		cpu_TotalN[i+1]=cpu_TotalN[i]/IntParams[2+i*3]; //N2*N3, N3...
		cpu_LeftLim[i]=IntParams[i*3]; //левый предел
		//printf("LeftLim %f \n",cpu_LeftLim[i]);
	}
	
    cudaMemcpy( dev_dX, cpu_dX,  multiplicity*sizeof(FLOAT), cudaMemcpyDefault); //копирует multiplicity*sizeof(FLOAT) байт из cpu_dX в dev_dX
    cudaMemcpy( dev_TotalN, cpu_TotalN,  (multiplicity+1)*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy( dev_LeftLim, cpu_LeftLim,  multiplicity*sizeof(FLOAT), cudaMemcpyDefault); 

    
    cudaMalloc(&dev_IntParams, multiplicity*3*sizeof(FLOAT));
    cudaMalloc(&dev_FuncParam, NumParam*sizeof(cuFLOATComplex));
    
    cudaMemcpy( dev_IntParams, IntParams,  multiplicity*3*sizeof(FLOAT), cudaMemcpyDefault);
    cudaMemcpy( dev_FuncParam, FuncParam,  NumParam*sizeof(cuFLOATComplex), cudaMemcpyDefault);
    
    cudaMalloc(&dev_J, count*sizeof(FLOAT));
    cudaMemset(dev_J,0,count*sizeof(FLOAT)); //заполняет 1*sizeof(FLOAT) байт  в dev_Result значением 0


 //вызов ядра
 
	//printf("Kernel start");
 
	KernelParallelIntegrator<<<Nblocks, Nthreads>>>(dev_J, multiplicity, dev_IntParams, dev_FuncParam, dev_dX, dev_TotalN, dev_LeftLim);
	gpuErrchk( cudaPeekAtLastError() );
	//printf("Kernel finish");
	cudaDeviceSynchronize(); 

	//cudaDeviceSynchronize(); //синхронизация
    cudaMemcpy(J_res, dev_J,  count*sizeof(FLOAT), cudaMemcpyDefault);

cudaFree(dev_IntParams); //освобождение выделенной памяти
cudaFree(dev_FuncParam);
cudaFree(dev_J);
cudaFree(dev_LeftLim);

cudaFree(dev_TotalN);
delete cpu_TotalN;
cudaFree(dev_dX),
delete cpu_dX;
delete cpu_LeftLim;

}

