/*
Copyright (c) 2025 ANNENKOV Vladimir, VOLCHOK Evgeniia
for contacts annenkov.phys@gmail.com, e.p.volchok@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
*/

// Radiation current
// see https://doi.org/10.1088/1361-6587/abdcdb

__device__ void Current(FLOAT *X, FLOAT *FuncParam, cuComplex3 *J)
{	
	FLOAT sin_a = FuncParam[3], cos_a = FuncParam[4], // sin(alpha), cos(alpha)
	cos_2a = FuncParam[7], //cos(2 alpha)
	sin_a2 = FuncParam[5], cos_a2 = FuncParam[6], // sin(alpha)^2, cos(alpha)^2
	sin_phi = sin(FuncParam[2]), cos_phi = cos(FuncParam[2]), //синус и косинус фи
	sin_theta = sin(FuncParam[1]), cos_theta = cos(FuncParam[1]); //sin(theta), cos(theta)
	FLOAT k = co_FuncParam[4];
	FLOAT sig1_2; // sigma_1^2
	FLOAT sig2_2; // sigma_2^2
	FLOAT sig0_1_2 = co_FuncParam[2]; //sigma_01^2, sigma_02^2
	FLOAT sig0_2_2 = co_FuncParam[3];
	FLOAT R1 = co_FuncParam[0]; // Rayleigh length
	FLOAT R2 = co_FuncParam[1];
	
	FLOAT Revsig1_2, Revsig1_4; // 1/sigma_1^2, 1/sigma_1^4
	FLOAT Revsig2_2, Revsig2_4; // 1/sigma_2^2, 1/sigma_2^4
	FLOAT sigA;
	FLOAT x1, x2, x1_2, x2_2, z1, z2;
	cuFLOATComplex EXP;
	
	cuFLOATComplex I = make_cuFLOATComplex(0,1.);
	
	FLOAT x, y, z;
	
	x = X[0];
	y = X[1];
	z = X[2];
	x1 = x*cos_a - z*sin_a;
	z1 = x*sin_a + z*cos_a;
		
	x2 = x*cos_a + z*sin_a;
	z2 = x*sin_a - z*cos_a;
	
	#if DIFRACTION == 0
		sig1_2 = sig0_1_2;
		sig2_2 = sig0_2_2;
	#endif
		
	#if DIFRACTION == 1
		sig1_2 = sig0_1_2 * (1. + z1*z1/(R1*R1));
		sig2_2 = sig0_2_2 * (1. + z2*z2/(R2*R2));
	#endif
	
	x1_2 = x1*x1;
	x2_2 = x2*x2;
	
	Revsig1_2 = 1./sig1_2;
	Revsig1_4 = Revsig1_2*Revsig1_2;
	Revsig2_2 = 1./sig2_2;
	Revsig2_4 = Revsig2_2*Revsig2_2;
	sigA = sig0_1_2 * sig0_2_2 * Revsig1_2 * Revsig2_2;
	EXP = exp(-2. * Revsig1_2 * (x1_2 + y*y) - 2. * Revsig2_2 * (x2_2 + y*y) + \
		I * (2. * sin_a*x - k*x*cos_phi*sin_theta - k*y*sin_phi*sin_theta - k*z*cos_theta));
		
		
	(*J).x = sigA * EXP * \
	(sin_a * (cos_a2+1.) * (4.*x2_2*Revsig2_4 + 4.*x1_2*Revsig1_4 - 1.*Revsig2_2-1.*Revsig1_2) + \
	sin_a * (4.*y*y*Revsig2_4 + 4.*y*y*Revsig1_4 - 1.*Revsig2_2 - 1.*Revsig1_2) + \
	4. * (x1*x2*Revsig1_2*Revsig2_2) * (3. - 4.*sin_a2) * sin_a + \
	sin_a * 4. * y * y * Revsig1_2 * Revsig2_2 - \
	0.25 * (2.*sin_a2 + 1.) * sin_a - \
	I * cos_a * (cos_a2 + 0.5) * (4.*(x1 + x2)*Revsig1_2*Revsig2_2 - \
	16.*x1*x2*(x2*Revsig2_2 + x1*Revsig1_2)*Revsig1_2*Revsig2_2) - \
	I * cos_a * (4.*(x1 + x2)*Revsig1_2*Revsig2_2 - 16.*y*y*(x1*Revsig2_2 + x2*Revsig1_2)*Revsig1_2*Revsig2_2) + \
	0.5 * I * cos_a * 16. * y * y * (x1*Revsig1_2 + x2*Revsig2_2)*Revsig1_2*Revsig2_2 - \
	I * (x2*Revsig2_2 + x1*Revsig1_2) * (3.*sin_a2 + 0.5)*cos_a);
	
	
	(*J).y = sigA * EXP * \
	(sin_a * cos_a * (Revsig1_2 + Revsig2_2) * (x1*Revsig1_2 + x2*Revsig2_2) * 4. * y - \
	I * 8. *y * Revsig1_2 * Revsig2_2 * (1. - 2.*x1_2*Revsig1_2 - 2.*x2_2*Revsig2_2) - \
	12. * I * y * Revsig1_2 * Revsig2_2 * (1. - 2*y*y*Revsig1_2 - 2*y*y*Revsig2_2) + \
	8. * I * cos_2a * x1 * x2 * y * Revsig1_2 * Revsig2_2 * (Revsig1_2 + Revsig2_2) - \
	I * (1. - cos_2a*0.5) * y * (Revsig2_2 + Revsig1_2));
	
	(*J).z = sigA * EXP * \
	(cos_a*(1. + sin_a2) * (Revsig1_2 - Revsig2_2 + 4.*(x2_2*Revsig2_4 - x1_2*Revsig1_4)) + \
	cos_a * (Revsig1_2 - Revsig2_2 + 4.*y*y*(Revsig2_4 - Revsig1_4)) + \
	I * sin_a * (1. - 0.5*cos_2a) * (x1*Revsig1_2 - x2*Revsig2_2) - \
	I * sin_a * (1. - 0.5*cos_2a) * 4. * Revsig1_2 * Revsig2_2 * (x2 - x1 - 4.*x1_2*x2*Revsig1_2 + 4.*x2_2*x1*Revsig2_2) - \
	I * sin_a * 4. * Revsig1_2 * Revsig2_2 * (x2 - x1 - 4.*y*y*x2*Revsig1_2 + 4.*y*y*x1*Revsig2_2) - \
	8. * I * sin_a * y * y * Revsig1_2 * Revsig2_2 * (x1*Revsig1_2 - x2*Revsig2_2));
	
}

