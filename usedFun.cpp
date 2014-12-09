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
