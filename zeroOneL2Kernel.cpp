#include "zeroOneL2Kernel.hpp"
#include "usedFun.hpp"

zeroOneL2Kernel::zeroOneL2Kernel(matd &data, unsigned int hidden):_hidden(hidden){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).stableNorm();
  }
  _norm = 0;
  _preCalc.resize(_hidden);
  vector<unsigned int> multi(4);
  for(size_t s1=0;s1<_hidden;++s1){
    _preCalc[s1].resize(_hidden);
    for(size_t s2=0;s2<_hidden;++s2){
      size_t start = (s1+s2-_hidden >0)? s1+s2-_hidden:0;
      size_t end = (s1<s2)? s1:s2;
      _preCalc[s1][s2].resize(_hidden);
      for(size_t i=start; i<=end;++i){
	multi[0] = i;
	multi[1] = s1-i;
	multi[2] = s2-i;
	multi[3] = _hidden-s1-s2+i;
	_preCalc[s1][s2][i] = (1-OneDpi*acos(i/sqrt(s1*s2)))*multinomial(_hidden,multi);
	_norm += _preCalc[s1][s2][i];
      }
    }
  }
}
    
double zeroOneL2Kernel::squaredNorm(size_t i){
  return _norm;
}
void zeroOneL2Kernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd alpha = OneDpi* ((((_data.transpose()*_data.col(i)).array()/
			     (_dataNorm[i] * _dataNorm)).min(1)).max(-1)).acos();
  res.setZero();
  for(size_t s1=0;s1<_hidden;++s1){
    for(size_t s2=0;s2<_hidden;++s2){
      size_t start = (s1+s2-_hidden >0)? s1+s2-_hidden:0;
      size_t end = (s1<s2)? s1:s2;
      for(size_t i=start; i<=end;++i){
	res = res.array() + _preCalc[s1][s2][i] * (1-alpha).pow(_hidden-s1-s2+2*i)*alpha.pow(s1+s2-2*i);
      }
    }
  }	
}
void zeroOneL2Kernel::dot(vec &v,Ref<VectorXd> res){

}
void zeroOneL2Kernel::dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res){
}
size_t zeroOneL2Kernel::getN(){
  return _n;
}
string zeroOneL2Kernel::getName(){
  char buffer[30];
  sprintf(buffer,"zeroOneL2Kernel\t%i",_hidden);
  string st(buffer);
  return st;
}
