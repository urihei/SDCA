#ifndef _usedFun__
#define _usedFun__

#include "def.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

void initRand();
int roll(unsigned int d);
void randperm(unsigned int n,ivec &arr,ivec &prePrm);
void cumsum(double* a, size_t len,double* b);
size_t findFirstBetween(double* a,double* bigger, size_t len);
size_t findFirst(double* a,size_t len);
void fillMatrix(matd  &data1, mat &data2);
unsigned long int prod(unsigned int s, unsigned int e);
double sumLog(unsigned int s, unsigned int e);
unsigned long long int multinomial(vector <unsigned int> &v);
double logMultinomial(vector <unsigned int> &v);
void ReadTrainData(string fileName,matd& data,ivec & label,vector<int> & label_map);
double AddNormAsFeature(matd &data);
double calcNormFeature(double squeredNorm,double max_norm);
double normalizeData(matd &data, vec &meanVec);
void normalizeData(matd &data, vec &meanVec, double maxNorm);

#endif
