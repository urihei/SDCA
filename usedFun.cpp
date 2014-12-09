#include "usedFun.hpp"
boost::mt19937 gen(time(NULL));

int roll(unsigned int d){
   boost::random::uniform_int_distribution<> dist(0, d);
   return  dist(gen);
}
    
void randperm(unsigned int n,unsigned int* arr){
    arr[0] = 0;
    for(size_t i =1; i<n;i++){
        unsigned int d = roll(i);
        arr[i] = arr[d];
        arr[d] = i;
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
size_t findFirstBetween(double* a,double* bigger,double* smaller, size_t len){
    if(len == 0)
        return -1;
    
    size_t i = 0;
    while((i<len)&&((a[i] < bigger[i]) || (a[i] > smaller[i])))
        i++;
    return i;
}
size_t findLast(double* a,size_t len){
   if(len == 0)
        return -1;
   len--;
   while((len>=0)&& (a[len]>0))
       len--;
   return len;
}
