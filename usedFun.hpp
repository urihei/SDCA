#ifndef _usedFun__
#define _usedFun__
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

int roll(unsigned int d);
void randperm(unsigned int n,unsigned int* arr);
void cumsum(double* a, size_t len,double* b);
size_t findFirstBetween(double* a,double* bigger, size_t len);
size_t findFirst(double* a,size_t len);
#endif
