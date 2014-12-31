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
double rbfKernel::squaredNorm(size_t i){
  return 1.0;
}
void rbfKernel::dot(size_t i,VectorXd & res){
  res =  exp((2*(_data.transpose()*_data.col(i)).array() - _dataSquare[i] - _dataSquare).transpose()/_sigma);

}
void rbfKernel::dot(vec &v,VectorXd &res){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  res = exp((2*(_data.transpose()*tmp).array()-squareNormData-_dataSquare)/_sigma);
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
