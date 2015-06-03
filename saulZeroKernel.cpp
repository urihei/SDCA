#include "saulZeroKernel.hpp"
#include "usedFun.hpp"

saulZeroKernel::saulZeroKernel(double* data,size_t n,size_t p, unsigned int l):_p(p),_l(l),_data(data,p,n){
  _n = n;
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).stableNorm();
  }
}
double saulZeroKernel::calc(double angle, unsigned int l){
  if(l>0){
    return dot(angle,l-1);
  }
  return 1-OneDpi*acos(angle);
}

double saulZeroKernel::dot(size_t i, size_t j){
  double angle = _data.col(j).dot(_data.col(i))/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(angle,_l);
}

double saulZeroKernel::dot(vec &v, size_t j){
  return dot(v.data(),j);
}
double saulZeroKernel::dot(double* v,size_t j){
  Map<VectorXd> vm(v,_p,1);
  double vmNorm = vm.stableNorm();
  double angle = vm.dot(_data.col(j))/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(angle,_l);
}

double saulZeroKernel::squaredNorm(size_t i){
  return 1;
}

void saulZeroKernel::dot(Ref<ArrayXd> alpha, unsigned int l){
  if(l>0){
    dot(alpha,l-1);
  }
  alpha = 1-OneDpi*alpha.acos();
}
void saulZeroKernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd preAn = (((_data.col(i).transpose()*_data).array()/
                    (_dataNorm[i] * _dataNorm.transpose())).min(1)).max(-1);
  dot(preAn,_l);
  res =  preAn;
}
void saulZeroKernel::dot(vec &v,Ref<VectorXd>res){
  dot(v.data(),res);
}
void saulZeroKernel::dot(double* v,Ref<VectorXd> res){
  Map<VectorXd> vm(v,_p,1);
  ArrayXd preAn = (((vm.transpose()*_data).array()/
                    (vm.stableNorm() * _dataNorm.transpose())).min(1)).max(-1);
  dot(preAn,_l);
  res =  preAn;
}

void saulZeroKernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
  ArrayXd preAn = (((v.transpose()*_data).array()/
                    (v.stableNorm() * _dataNorm.transpose())).min(1)).max(-1);
  dot(preAn,_l);
  res =  preAn;
}
size_t saulZeroKernel::getN(){
  return _n;
}
string saulZeroKernel::getName(){
  char buffer[30];
  sprintf(buffer,"saulZero\t%16i",_l);
  string st(buffer);
  return st;
}
