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
  _norm = pow(2,_hidden)-1;

  _preCalc.resize(_hidden+1);
  for(size_t i = 0;i<=_hidden;++i){
    _preCalc[i] =0;
  }
  vector<unsigned int> multi(4);
  for(size_t s1=1;s1<=_hidden;++s1){
    for(size_t s2=1;s2<=_hidden;++s2){
      size_t start = (s1+s2 >_hidden)? s1+s2-_hidden:0;
      size_t end = (s1<s2)? s1:s2;
      for(size_t i=start; i<=end;++i){
        multi[0] = i;
        multi[1] = s1-i;
        multi[2] = s2-i;
        multi[3] = _hidden+i-s1-s2;
        //When _hidden is large use stable version otherwise use more accurate
        if(_hidden <= 40){
          _preCalc[s1+s2-2*i] +=(1-OneDpi*acos(i/sqrt(s1*s2)))*multinomial(multi);
        }else{
          double tmp  = log((1-OneDpi*acos(i/sqrt(s1*s2))))+logMultinomial(multi);
          if(tmp>_preCalc[s1+s2-2*i]){
            _preCalc[s1+s2-2*i] = tmp + log(1+exp(_preCalc[s1+s2-2*i]-tmp));
          }else{
            _preCalc[s1+s2-2*i] = _preCalc[s1+s2-2*i] +
              log(1+exp(tmp -_preCalc[s1+s2-2*i]));
          }
        }
                    
        //                cerr<<_preCalc[s1-1][s2-1][i]<<" ";
      }
      //            cerr<<endl;
    }
  }
  if(_hidden <= 40){
    for(size_t i=0; i<=_hidden;++i)
      _preCalc[i] = log(_preCalc[i]);
  }
  //
  // _preCalc.resize(2*_hidden);
  // for(size_t c=2;c<=2*_hidden;++c){
  //     _preCalc[c-2].resize(_hidden+1);
  //     size_t start_i = (c >_hidden)? c-_hidden:0;
  //     for(size_t i=start_i; i<=c/2;++i){
  //         _preCalc[c-2][i] = 0;
  //         size_t start_s = (i > 0)? i:1;
  //         size_t end_s = (i==0)? c-1:c-i;
  //         end_s = (end_s < _hidden)? end_s:_hidden;
  //         for(size_t s=start_s ;s<= end_s;++s){
  //             multi[0] = i;
  //             multi[1] = s-i;
  //             multi[2] = c-s-i;
  //             multi[3] = _hidden+i-c;
  //             _preCalc[c-2][i] += (1-OneDpi*acos(i/sqrt(s*(c-s))))*multinomial(multi);
  //         }
  //     }
  // }
  //
}
double zeroOneL2Kernel::calc(double alpha){
  double res = 0;
  alpha = (alpha < eps)? eps:alpha;
  alpha = (alpha > 1-eps)? 1-eps:alpha;

  double lAlpha = log(alpha);
  double oMa = log(1 - alpha);
  for(size_t i=0;i<=_hidden;++i){
    res += exp(_preCalc[i]+i*lAlpha+(_hidden-i)*oMa);
  }
  return res /_norm;
}
double zeroOneL2Kernel::dot(size_t i, size_t j){
  double angle = _data.col(j).dot(_data.col(i))/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(OneDpi*acos(angle));
}

double zeroOneL2Kernel::dot(vec &v, size_t j){
  Map<VectorXd> vm(v.data(),_n,1);
  double vmNorm = vm.stableNorm();
  double angle = vm.dot(_data.col(j))/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(OneDpi*acos(angle));
}

double zeroOneL2Kernel::squaredNorm(size_t i){
  return 1;
}
void zeroOneL2Kernel::calc(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res){
  res.setZero();
  ArrayXd tmpAlpha = alpha.max(eps).min(1-eps).log();//(alpha.size());
  //tmpAlpha.setOnes();
  //  tmpAlpha = 
  ArrayXd tmpOneMinusAlpha(alpha.size());
  tmpOneMinusAlpha = (1-alpha.max(eps).min(1-eps)).log();
  for(size_t i=0;i<=_hidden;++i){
    res = res.array() + (_preCalc[i]+i*tmpAlpha+(_hidden-i)*tmpOneMinusAlpha).exp();
  }
  res = res /_norm;
}
/*
  void zeroOneL2Kernel::calc2(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res){
  res.setZero();
  for(size_t s1=1;s1<=_hidden;++s1){
    for(size_t s2=1;s2<=_hidden;++s2){
      size_t start = (s1+s2 >_hidden)? s1+s2-_hidden:0;
      size_t end = (s1<s2)? s1:s2;
      for(size_t i=start; i<=end;++i){
        res = res.array() + _preCalc[s1-1][s2-1][i] * (1-alpha).pow(_hidden-s1-s2+2*i)*alpha.pow(s1+s2-2*i);
      }
    }
  }
  res = res /_norm;
}
*/
void zeroOneL2Kernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd alpha = OneDpi* ((((_data.transpose()*_data.col(i)).array()/
                             (_dataNorm[i] * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res);
}
void zeroOneL2Kernel::dot(vec &v,Ref<VectorXd> res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  normVec = sqrt(normVec);
  ArrayXd alpha = OneDpi*((((_data.transpose()*tmp).array()/
                            (normVec * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res);
}
void zeroOneL2Kernel::dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res){
  ArrayXd alpha = OneDpi*((((_data.transpose()*v).array()/
                            (v.stableNorm() * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res);
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
