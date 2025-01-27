
__device__ FLOAT cusincf(FLOAT *x,FLOAT *param)
{
	return sin(param[0]*x[0])/(param[1]*x[0]);
}

__device__ FLOAT func2(FLOAT *x,FLOAT *param)
{
	return exp(-x[0]*x[0]/(param[0]*param[0])-x[1]*x[1]/(param[1]*param[1])-x[2]*x[2]/(param[2]*param[2]));
}

__device__ FLOAT co_func2(FLOAT *x,FLOAT *FuncParam)
{

	return exp(-(x[0]-FuncParam[0])*(x[0]-FuncParam[0])/(FuncParam[0]*FuncParam[0])-x[1]*x[1]/(FuncParam[1]*FuncParam[1])-x[2]*x[2]/(FuncParam[2]*FuncParam[2]));
}

__device__ cuFLOATComplex expT(FLOAT *x,cuFLOATComplex *FuncParam)
{
	//printf("%d, 	%f \n", sizeof(FuncParam), FuncParam[0]);
	return FuncParam[0]*exp(-x[0]*x[0]*FuncParam[1]);
}

__device__ FLOAT co_expT(FLOAT *x,FLOAT *FuncParam)
{
	//printf("%d, 	%f \n", sizeof(FuncParam), FuncParam[0]);
	return FuncParam[0]*exp(-x[0]*x[0]);
}

__device__ FLOAT FuncSwitch(FLOAT *x,FLOAT *param,int Func)
{
	FLOAT res;

	switch(Func)
	{
	case(0):
		res=cusincf(x,param);
	break;
	case(1):
		res=func2(x,param);
	break;
	
	case(101)://функция, использующая константную память
		res=co_func2(x,param);
	break;
	}
	return res;
}

__device__ void Current(FLOAT *X, FLOAT *FuncParam, cuComplex3 *J)
{	
	FLOAT sin_a=FuncParam[3],cos_a=FuncParam[4],//синус и косинус альфа
	cos_2a=FuncParam[7],//косинус 2альфа
	sin_a2=FuncParam[5],cos_a2=FuncParam[6],//синус и косинус в квадрате от альфы
	sin_phi=sin(FuncParam[2]),cos_phi=cos(FuncParam[2]),//синус и косинус фи
	sin_theta=sin(FuncParam[1]),cos_theta=cos(FuncParam[1]);//синус и косинус тета
	FLOAT k=co_FuncParam[4];
	FLOAT sig1_2;//сигма 1 в квадрате
	FLOAT sig2_2;//сигма 2 в квадрате 
	FLOAT sig0_1_2=co_FuncParam[2];//сигмы нулевые в квадрате
	FLOAT sig0_2_2=co_FuncParam[3];
	FLOAT R1=co_FuncParam[0];//длины релея
	FLOAT R2=co_FuncParam[1];
	
	FLOAT Revsig1_2,Revsig1_4;//сигма 1 в квадрате и в четвёртой всё в минус первой
	FLOAT Revsig2_2,Revsig2_4;//сигма 2 в квадрате и в четвёртой всё в минус первой
	FLOAT sigA;
	FLOAT x1,x2,x1_2,x2_2,z1,z2;
	cuFLOATComplex EXP;
	
	cuFLOATComplex I=make_cuFLOATComplex(0,1.);
	
	FLOAT x,y,z;
	
	x=X[0];
	y=X[1];
	z=X[2];
	x1=x*cos_a-z*sin_a;
	z1=x*sin_a+z*cos_a;
		
	x2=x*cos_a+z*sin_a;
	z2=x*sin_a-z*cos_a;
	
	#if DIFRACTION==0
		sig1_2=sig0_1_2;
		sig2_2=sig0_2_2;
	#endif
		
	#if DIFRACTION==1

		sig1_2=sig0_1_2*(1.+z1*z1/(R1*R1));
		sig2_2=sig0_2_2*(1.+z2*z2/(R2*R2));
	#endif
	
	x1_2=x1*x1;
	x2_2=x2*x2;
	
	//(*J).x=make_cuFLOATComplex(1.,1.);
	
	Revsig1_2=1./sig1_2;
	Revsig1_4=Revsig1_2*Revsig1_2;
	Revsig2_2=1./sig2_2;
	Revsig2_4=Revsig2_2*Revsig2_2;
	sigA=sig0_1_2*sig0_2_2*Revsig1_2*Revsig2_2;
	EXP=exp(-2.*Revsig1_2*(x1_2+y*y)-2.*Revsig2_2*(x2_2+y*y)+I*(2.*sin_a*x-k*x*cos_phi*sin_theta-k*y*sin_phi*sin_theta-k*z*cos_theta));
		
		
	(*J).x=sigA*EXP*(sin_a*(cos_a2+1.)*(4.*x2_2*Revsig2_4+4.*x1_2*Revsig1_4-1.*Revsig2_2-1.*Revsig1_2)+sin_a*(4.*y*y*Revsig2_4+4.*y*y*Revsig1_4-1.*Revsig2_2-1.*Revsig1_2)+4.*(x1*x2*Revsig1_2*Revsig2_2)*(3.-4.*sin_a2)*sin_a+sin_a*4.*y*y*Revsig1_2*Revsig2_2-0.25*(2.*sin_a2+1.)*sin_a-I*cos_a*(cos_a2+0.5)*(4.*(x1+x2)*Revsig1_2*Revsig2_2-16.*x1*x2*(x2*Revsig2_2+x1*Revsig1_2)*Revsig1_2*Revsig2_2)-I*cos_a*(4.*(x1+x2)*Revsig1_2*Revsig2_2-16.*y*y*(x1*Revsig2_2+x2*Revsig1_2)*Revsig1_2*Revsig2_2)+0.5*I*cos_a*16.*y*y*(x1*Revsig1_2+x2*Revsig2_2)*Revsig1_2*Revsig2_2-I*(x2*Revsig2_2+x1*Revsig1_2)*(3.*sin_a2+0.5)*cos_a);
	
	
	(*J).y=sigA*EXP*(sin_a*cos_a*(Revsig1_2+Revsig2_2)*(x1*Revsig1_2+x2*Revsig2_2)*4.*y-I*8.*y*Revsig1_2*Revsig2_2*(1.-2.*x1_2*Revsig1_2-2.*x2_2*Revsig2_2)-12.*I*y*Revsig1_2*Revsig2_2*(1.-2*y*y*Revsig1_2-2*y*y*Revsig2_2)+8.*I*cos_2a*x1*x2*y*Revsig1_2*Revsig2_2*(Revsig1_2+Revsig2_2)-I*(1.-cos_2a*0.5)*y*(Revsig2_2+Revsig1_2));
	
	(*J).z=sigA*EXP*(cos_a*(1.+sin_a2)*(Revsig1_2-Revsig2_2+4.*(x2_2*Revsig2_4-x1_2*Revsig1_4))+cos_a*(Revsig1_2-Revsig2_2+4.*y*y*(Revsig2_4-Revsig1_4))+I*sin_a*(1.-0.5*cos_2a)*(x1*Revsig1_2-x2*Revsig2_2)-I*sin_a*(1.-0.5*cos_2a)*4.*Revsig1_2*Revsig2_2*(x2-x1-4.*x1_2*x2*Revsig1_2+4.*x2_2*x1*Revsig2_2)-I*sin_a*4.*Revsig1_2*Revsig2_2*(x2-x1-4.*y*y*x2*Revsig1_2+4.*y*y*x1*Revsig2_2)-8.*I*sin_a*y*y*Revsig1_2*Revsig2_2*(x1*Revsig1_2-x2*Revsig2_2));
	
}

