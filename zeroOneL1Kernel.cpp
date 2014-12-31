#include "zeroOneL1Kernel.hpp"
#include "usedFun.hpp"

zeroOneL1Kernel::zeroOneL1Kernel(matd &data){
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataSquare.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).norm();
  }
}
double zeroOneL1Kernel::squaredNorm(size_t i){
  return 1;
}
void zeroOneL1Kernel::dot(size_t i,VectorXd & res){
  //  cerr<<"R "<<(_data.transpose()*_data.col(i)).rows()<<" C "<<(_data.transpose()*_data.col(i)).cols()<<"<-> R "<<(_dataSquare[i] * _dataSquare).rows()<<" C "<<(_dataSquare[i] * _dataSquare).cols()<<endl;

  res =  1-OneDpi*acos((_data.transpose()*_data.col(i)).array()/
		       (_dataSquare[i] * _dataSquare));
}
void zeroOneL1Kernel::dot(vec &v,VectorXd &res){
  VectorXd tmp(_p);
  double squareNormData = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    squareNormData += v[i]*v[i];
  }
  res =  1-OneDpi*acos((_data.transpose()*tmp).array()/
		       (squareNormData * _dataSquare));
}
size_t zeroOneL1Kernel::getN(){
  return _data.cols();
}
