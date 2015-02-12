#include "rbfKernel.hpp"
#include "usedFun.hpp"

rbfKernel::rbfKernel(matd &data,double sigma):
_sigma(2*sigma){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataSquare.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataSquare(i) = _data.col(i).squaredNorm();
  }
}
double rbfKernel::dot(size_t i, size_t j){
  return exp((2*((_data.col(j)).dot(_data.col(i))) - _dataSquare[i] - _dataSquare[j])/_sigma);
}
double rbfKernel::dot(vec &v, size_t j){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  return exp((2*((_data.col(j)).dot(tmp)) - squareNormData - _dataSquare[j])/_sigma);
}
 

double rbfKernel::squaredNorm(size_t i){
  return 1.0;
}
void rbfKernel::dot(size_t i,Ref<VectorXd>  res){
  res =  exp((2*(_data.transpose()*_data.col(i)).array() - _dataSquare[i] - _dataSquare).transpose()/_sigma);

}
void rbfKernel::dot(vec &v,Ref<VectorXd> res){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  res = exp((2*(_data.transpose()*tmp).array()-squareNormData-_dataSquare)/_sigma);
}

void rbfKernel::dot(const Ref<const VectorXd> &v,Ref<VectorXd> res){
    res = exp((2*(_data.transpose()*v).array()-v.squaredNorm()-_dataSquare)/_sigma);
}
size_t rbfKernel::getN(){
  return _data.cols();
}
void rbfKernel::setSigma(double sigma){
  _sigma= sigma;
}
double rbfKernel::getSigma(){
  return _sigma;
}
string rbfKernel::getName(){
    char buffer[30];
    sprintf(buffer,"RBF\t%16g",_sigma);
    string st(buffer);
    return st;
}
