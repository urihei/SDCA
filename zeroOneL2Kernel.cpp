#include "zeroOneL2Kernel.hpp"


zeroOneL2Kernel::zeroOneL2Kernel(matd &data, unsigned int hidden):_hidden(hidden){
    fillMatrix(data,_data);
    _data.transposeInPlace();
    _p = _data.rows();
    _n = _data.cols();
    _dataNorm.resize(_n);
    for(size_t i=0; i<_n;++i){
        _dataNorm(i) = _data.col(i).stableNorm();
    }
    _preCalc.resize(_hidden);
    for(size_t s1=0;s1<_hidden;++s1){
        _preCalc[s1].resize(_hidden);
        for(size_t s2=0;s2<_hidden;++s2){
            size_t start = (s1+s2-_hidden >0)? s1+s2-_hidden:0;
            size_t end = (s1<s2)? s1:s2;
            for(size_t i=start; i<=end;++i){
                _preCalc[s1][s2][i] = (1-OneDpi*acos(i/sqrt(s1*s2)))*multinomial(_hidden,i,s1-i,s2-i,_hidden-s1-s2+i);
            }
        }
    }
}
    
virtual double squaredNorm(size_t i);
virtual void dot(size_t i,Ref<VectorXd> res);
virtual void dot(vec &v,Ref<VectorXd> res);
virtual void dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res);
virtual size_t getN();
virtual string getName();
