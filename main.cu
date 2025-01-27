#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>  //Для clock_gettime
#include <fstream>
#include <iostream>
#include <complex>
#include <stdlib.h>
#include <unistd.h>
//#include <omp.h>

#define REDUCTIONTYPE 1//0-mutex 1-параллельно
#define DIFRACTION 1
#define CompCount 6 //комплексный вектор xyz  3-Re, 3-Im
#define MAXCONSTPARAM 5

#define FLOATTYPE 0//0-FLOAT, 1-float

#define NBLOCKS 80
#define NTHREADS 96



using namespace std;
#include "LibIntGPU.cu"



int main()
{
	SetDevice(0);

	struct timespec mt1, mt2; 
	long int tt;      
	ofstream FileRes;
	ofstream FileTempRes;
	
	struct stat st={0};
	if(stat("./Results", &st)==-1)
		mkdir("./Results", 0700);
	
	FileRes.open("./Results/ResPower",std::ios::out);
	
	int N_theta=300;
	int N_phi=100;
	
	FLOAT d_theta=M_PI/FLOAT(N_theta);
	FLOAT d_phi=2.*M_PI/FLOAT(N_phi);
	
	FLOAT Alpha=0, theta=0, phi=0;
	
	FLOAT Sigma0_1=1.86,Sigma0_2=5.28;
	FLOAT w0=25.46;
	FLOAT E1=0.1775;
	FLOAT E2=0.2531;
	FLOAT k=sqrt(3);
	FLOAT eps=3./4.;
	FLOAT R1=w0*Sigma0_1*Sigma0_1*0.5, R2=w0*Sigma0_2*Sigma0_2*0.5;//длины релея
	
	FLOAT R=4.*Sigma0_1;
	FLOAT z0=150.;

	FLOAT IntParams[9]={-R, R, 100, -R, R, 100, -z0, z0, 1500};
	
	int NumParam=8; //число переменных параметров
    FLOAT FuncParam[8]={Alpha, theta, phi, sin(Alpha), cos(Alpha), sin(Alpha)*sin(Alpha), cos(Alpha)*cos(Alpha), cos(2.*Alpha)};
    
   // int NumCoParam=MAXCONSTPARAM; //число постоянных параметров
    FLOAT co_FuncParam[5]={R1, R2, Sigma0_1*Sigma0_1, Sigma0_2*Sigma0_2, k};
    CopyToConstantMem(MAXCONSTPARAM, co_FuncParam);
    
    int count=CompCount;
	FLOAT *J_res;
	J_res=new FLOAT[count];
    
	FLOAT *B;
	B=new FLOAT[count];
	
	for (int i=0; i<count; i++)
	{	J_res[i]=0;
		B[i]=0;
	}
    
	FLOAT Int=0.;
	FLOAT tempPower=0.;
	FLOAT Power=0;
	FLOAT PowerCoeff=0.69/(8.*sqrt(eps)*M_PI*M_PI);
	FLOAT Coeff=E1*E2*k;
	
	ostringstream strs;
	string str;
	str="";
	FileRes<<"Alpha	Power	time(sec)"<<endl;
	for(Alpha=0; Alpha<M_PI_2; Alpha+=0.087266463) //шаг 5 градусов
	{
		printf("Alpha %f \n", Alpha);
		
		clock_gettime (CLOCK_REALTIME, &mt1);
		
		FuncParam[0]=Alpha;
		FuncParam[3]=sin(Alpha);
		FuncParam[4]=cos(Alpha);
		FuncParam[5]=sin(Alpha)*sin(Alpha);
		FuncParam[6]=cos(Alpha)*cos(Alpha);
		FuncParam[7]=cos(2.*Alpha);
		
		strs<<Alpha/M_PI*180.;
		str = strs.str();
		FileTempRes.open(("./Results/ResPowerAlpha0"+str).c_str(),std::ios::out);
		strs.str("");
		strs.clear();
		
		FileTempRes<<"i_theta"<<"	"<<"theta"<<"	"<<"i_phi"<<"	"<<"phi"<<"	"<<"Bx.real()"<<"	"<<"Bx.imag()"<<"	"<<"By.real()"<<"	"<<"By.imag()"<<" "<<"Bz.real()"<<"	"<<"Bz.imag()"<<"	"<<"Int"<<"	"<<"Int*PowerCoeff"<<endl;
	
		for(int i_theta=0;i_theta<N_theta;i_theta++)
		{
			theta=i_theta*d_theta+0.5*d_theta;
			FuncParam[1]=theta;
			
			for(int i_phi=0; i_phi<N_phi; i_phi++) 
			{	
				phi=i_phi*d_phi+0.5*d_phi-M_PI_2;
				FuncParam[2]=phi;
				
				ParallelNquadIntegrator(3,IntParams, NumParam, FuncParam, J_res);
				printf("i %d, j %d, theta %f, phi %f, Re(J_res.x) %f, Im(J_res.x) %f, Re(J_res.y) %f, Im(J_res.y) %f, Re(J_res.z) %f, Im(J_res.z) %f \n", i_theta, i_phi, theta, phi, J_res[0], J_res[0+3], J_res[1], J_res[1+3], J_res[2], J_res[2+3]);
				
				B[0]=(J_res[2]*sin(phi)*sin(theta)-J_res[1]*cos(theta))*Coeff; //Bx real
				B[1]=-(J_res[0]*cos(theta)-J_res[2]*cos(phi)*sin(theta))*Coeff; //By real
				B[2]=(J_res[1]*cos(phi)*sin(theta)-J_res[0]*sin(phi)*sin(theta))*Coeff; //Bz real
				
				B[0+3]=(J_res[2+3]*sin(phi)*sin(theta)-J_res[1+3]*cos(theta))*Coeff; //Bx imag
				B[1+3]=-(J_res[0+3]*cos(theta)-J_res[2+3]*cos(phi)*sin(theta))*Coeff; //By imag
				B[2+3]=(J_res[1+3]*cos(phi)*sin(theta)-J_res[0+3]*sin(phi)*sin(theta))*Coeff; //Bz imag
				
				
				Int=B[0]*B[0]+B[1]*B[1]+B[2]*B[2]+B[0+3]*B[0+3]+B[1+3]*B[1+3]+B[2+3]*B[2+3];
				tempPower=Int*PowerCoeff*sin(theta)*d_theta*d_phi;
				Power+=tempPower;
			
				FileTempRes<<i_theta<<"	"<<theta<<"	"<<i_phi<<"	"<<phi<<"	"<<B[0]<<"	"<<B[0+3]<<"	"<<B[1]<<" "<<B[1+3]<<"	"<<B[2]<<"	"<<B[2+3]<<"	"<<Int<<"	"<<tempPower<<"	"<<Power<<endl;
			}
			
		}
		
		cout<<"angle="<<Alpha/M_PI*180.<<"	"<<"Power="<<Power<<endl;

		
		clock_gettime (CLOCK_REALTIME, &mt2);

		//Рассчитываем разницу времени между двумя измерениями
		tt=1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);
		FileRes<<Alpha/M_PI*180.<<"	"<<Power<<"	"<<tt/1000000000.<<endl;
		Power=0;

		//Выводим результат расчета на экран
		printf ("time: %g с\n",tt/1000000000.);
		
		FileTempRes.close();
	}
	


	FileRes.close();


	delete J_res;
	delete B;
	
	return 0;
	

}

