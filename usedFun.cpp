#include "usedFun.hpp"
boost::mt19937 gen(time(NULL));//999);//

int roll(unsigned int d){
  boost::random::uniform_int_distribution<> dist(0, d);
  return  dist(gen);
}
    
void randperm(unsigned int n,ivec &arr, ivec &prePrm){
  arr[0] = prePrm[0];
  for(size_t i =1; i<n;i++){
    unsigned int d = roll(i);
    arr[i] = arr[d];
    arr[d] = prePrm[i];
  }
}
void cumsum(double* a, size_t len,double* b){
  if(len == 0)
    return;
  b[0] = a[0];
  for(size_t i=1;i<len;++i){
    b[i]=b[i-1]+a[i];
  }
}
/**
 * Compare the last element between com[len-1]<= a[len-1] <= 1.
 **/
size_t findFirstBetween(double* a,double* com, size_t len){
  if(len == 0)
    return -1;
    
  size_t i = 0;
  while((i<len-1)&&((a[i] < com[i]) || (a[i] > com[i+1])))
    i++;
  if(i==len-1){
    if((a[i]>=com[i])&& (a[i]<= 1)){
      return i;
    }
    return len;
  }
  return i;
}
size_t findFirst(double* a,size_t len){
  if(len == 0)
    return -1;
  size_t i=0;
  while((i<len) && a[i] != 1) 
    i++;
  return i;
}

void fillMatrix(matd &data1, mat &data2){
  size_t n = data1.size();
  if(n == 0)
      return;
  size_t p =  data1[0].size();
    
  data2.resize(n,p);
  for(size_t i=0;i<n;++i){
    for(size_t j=0;j<p;++j){
      data2(i,j) = data1[i][j];
    }
  }
}
