#ifndef _usedFun__
#define _usedFun__

#include "def.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

int roll(unsigned int d);
void randperm(unsigned int n,ivec &arr,ivec &prePrm);
void cumsum(double* a, size_t len,double* b);
size_t findFirstBetween(double* a,double* bigger, size_t len);
size_t findFirst(double* a,size_t len);
void fillMatrix(matd  &data1, mat &data2);
unsigned int prod(unsigned int s, unsigned int e);
unsigned int multinomial(unsigned int N,vector <unsigned int> &v);
#endif
