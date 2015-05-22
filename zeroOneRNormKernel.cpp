#include "zeroOneRNormKernel.hpp"
#include "usedFun.hpp"

zeroOneRNormKernel::zeroOneRNormKernel(matd &data, ivec & hidden):_l(hidden.size()){
  _hidden = hidden;
  fillMatrix(data,_data);
  _data.transposeInPlace();
  _p = _data.rows();
  _n = _data.cols();
  _dataNorm.resize(_n);
  _normFeature.resize(_n);
  _max_norm = -1;
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = _data.col(i).squaredNorm();
    _max_norm = (_dataNorm(i) > _max_norm)? _dataNorm(i):_max_norm;
  }
  for(size_t i=0; i<_n;++i){
    _normFeature(i) = calcNormFeature(_dataNorm(i),_max_norm);
    _dataNorm(i) = sqrt(_dataNorm(i)+_normFeature(i)*_normFeature(i));
  }
  
  _norm.resize(_l);
  for(size_t i=0;i<_l;++i){
    _norm[i] = pow(2,_hidden[i])-1;
  }
  _preCalc.resize(_l);
  for(size_t i = 0;i<_l;++i){
    _preCalc[i].resize(_hidden[i]+1);
    for(size_t j=0;j<=_hidden[i];++j){
      _preCalc[i][j] =0;
    }
  }

  vector<unsigned int> multi(4);
  for(size_t l=0;l<_l;++l){
    for(size_t s1=1;s1<=_hidden[l];++s1){
      for(size_t s2=1;s2<=_hidden[l];++s2){
        size_t start = (s1+s2 >_hidden[l])? s1+s2-_hidden[l]:0;
        size_t end = (s1<s2)? s1:s2;
        for(size_t i=start; i<=end;++i){
          multi[0] = i;
          multi[1] = s1-i;
          multi[2] = s2-i;
          multi[3] = _hidden[l]+i-s1-s2;
          //When _hidden is large use stable version otherwise use more accurate
          if(_hidden[l] <= 40){
            _preCalc[l][s1+s2-2*i] +=(1-OneDpi*acos(i/sqrt(s1*s2)))*multinomial(multi);
          }else{
            double tmp  = log((1-OneDpi*acos(i/sqrt(s1*s2))))+logMultinomial(multi);
            if(tmp>_preCalc[l][s1+s2-2*i]){
              _preCalc[l][s1+s2-2*i] = tmp + log(1+exp(_preCalc[l][s1+s2-2*i]-tmp));
            }else{
              _preCalc[l][s1+s2-2*i] = _preCalc[l][s1+s2-2*i] +
                log(1+exp(tmp -_preCalc[l][s1+s2-2*i]));
            }
          }
                    
        }
      }
    }
    if(_hidden[l] <= 40){
      for(size_t i=0; i<=_hidden[l];++i)
        _preCalc[l][i] = log(_preCalc[l][i]);
    }
  }
}
double zeroOneRNormKernel::calc(double alpha,unsigned int l){
  if(l==0){
    return 1-alpha;
  }
  alpha = calc(alpha,l-1);
  alpha = (alpha < eps)? eps:alpha;
  alpha = (alpha > 1-eps)? 1-eps:alpha;

  double lAlpha = log(alpha);
  double oMa = log(1 - alpha);
  double res = 0;
  for(size_t i=0;i<=_hidden[l];++i){
    res += exp(_preCalc[l][i]+i* oMa + (_hidden[l]-i)*lAlpha);
  }
  return res /_norm[l];
}
double zeroOneRNormKernel::dot(size_t i, size_t j){
  double angle = (_data.col(j).dot(_data.col(i))+_normFeature(i)*_normFeature(j))/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(OneDpi*acos(angle),_l);
}

double zeroOneRNormKernel::dot(vec & v, size_t j){
  Map<VectorXd> vm(v.data(),v.size(),1);
  double vmNorm = vm.squaredNorm();
  double normFeature = calcNormFeature(vmNorm,_max_norm);
  vmNorm = sqrt(vmNorm + normFeature*normFeature);
  double angle = (vm.dot(_data.col(j))+normFeature*_normFeature(j))/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(OneDpi*acos(angle),2);
}
    
double zeroOneRNormKernel::squaredNorm(size_t i){
  return 1;
}

void zeroOneRNormKernel::calc(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res,unsigned int l){
  if(l==0){
    res = 1-alpha;
    return;
  }
  calc(alpha,res,l-1);
  ArrayXd T = res.array().max(eps).min(1-eps).log();//(alpha.size());
  ArrayXd OneMinusT = (1-res.array().max(eps).min(1-eps)).log();
  res.setZero();
  for(size_t i=0;i<=_hidden[l];++i){
    res = res.array() + (_preCalc[l][i]+i* OneMinusT + (_hidden[l]-i)*T).exp();
  }
  res = res /_norm[l];
}
/*
  void zeroOneRNormKernel::calc2(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res){
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
void zeroOneRNormKernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd alpha = OneDpi* (((((_data.transpose()*_data.col(i)).array()+(_normFeature(i)*_normFeature))/
                             (_dataNorm(i) * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res,_l-1);
}

void zeroOneRNormKernel::dot(vec &v,Ref<VectorXd> res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p-1;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  double vecFeature = calcNormFeature(normVec,_max_norm);
  normVec = sqrt(normVec+vecFeature*vecFeature);
  ArrayXd alpha = OneDpi*(((((_data.transpose()*tmp).array()+(vecFeature * _normFeature))/
                            (normVec * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res,_l-1);
}
void zeroOneRNormKernel::dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res){
  double normVec = v.squaredNorm();
  double vecFeature = calcNormFeature(normVec,_max_norm);
  normVec = sqrt(normVec+vecFeature*vecFeature);
  ArrayXd alpha = OneDpi*(((((_data.transpose()*v).array()+(vecFeature*_normFeature))/
                            (v.stableNorm() * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res,_l-1);
}
size_t zeroOneRNormKernel::getN(){
  return _n;
}
string zeroOneRNormKernel::getName(){
  string st = "ZeroOneRNorm\t"+to_string(_l);
  for(size_t l=0;l<_l;++l){
    st += "\t"+to_string(_hidden[l]);
  }
  st += "\t"+to_string(_max_norm);
  return st;
}
