#include "zeroOneRBiasKernel.hpp"
#include "usedFun.hpp"

zeroOneRBiasKernel::zeroOneRBiasKernel(double* &data, size_t n, size_t p,ivec & hidden,vec &bias):_l(hidden.size()),_data(data,p,n),_p(p),_n(n){  
  _hidden = hidden;
  _dataNorm.resize(_n);
  for(size_t i=0; i<_n;++i){
    _dataNorm(i) = sqrt(_data.col(i).squaredNorm()+bias[0]*bias[0]);
  }
  _norm.resize(_l);
  _bias.resize(_l);
  cerr<<"Network:\tHidden\tBias"<<endl;
  for(size_t i=0;i<_l;++i){
    _norm[i] = pow(2,_hidden[i])-1;
    if(i >= bias.size()){
      _bias[i] = bias[0] *bias[0];
    }else{
      _bias[i] = bias[i]*bias[i];
    }
    if(i==0){
      cerr<<i<<":\t\t"<<"~"<<"\t"<<_bias[i]<<endl;
    }else{
      cerr<<i<<":\t\t"<<_hidden[i]<<"\t"<<_bias[i]<<endl;
    }
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
    // if(_hidden[l]<40){
    //   _preCalc[l][0]  = 2 * _hidden[l];
    // }else{
    //   _preCalc[l][0]  = log(2 * _hidden[l]);
    // }
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
            _preCalc[l][s1+s2-2*i] +=(1-OneDpi*acos((i+(_bias[l]))/sqrt((s1+_bias[l])*(s2+_bias[l]))))*multinomial(multi);
          }else{
            double tmp  = log(1-OneDpi*acos((i+_bias[l])/sqrt((s1+_bias[l])*(s2+_bias[l]))))+logMultinomial(multi);
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
double zeroOneRBiasKernel::calc(double alpha,unsigned int l){
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
double zeroOneRBiasKernel::dot(size_t i, size_t j){
  double angle = (_data.col(j).dot(_data.col(i))+_bias[0])/(_dataNorm(i) * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(OneDpi*acos(angle),_l);
}

double zeroOneRBiasKernel::dot(vec & v, size_t j){
  Map<VectorXd> vm(v.data(),_n,1);
  double vmNorm = sqrt(vm.squaredNorm()+_bias[0]);
  double angle = (vm.dot(_data.col(j))+_bias[0])/( vmNorm * _dataNorm(j));
  angle = (angle > 1)?  1:angle;
  angle = (angle <-1)? -1:angle;
  return calc(OneDpi*acos(angle),2);
}
    
double zeroOneRBiasKernel::squaredNorm(size_t i){
  return 1;
}

void zeroOneRBiasKernel::calc(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res,unsigned int l){
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
  void zeroOneRBiasKernel::calc2(const Ref<const ArrayXd> &alpha,Ref<VectorXd> res){
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
void zeroOneRBiasKernel::dot(size_t i,Ref<VectorXd> res){
  ArrayXd alpha = OneDpi* (((((_data.col(i).transpose() * _data).array()+_bias[0])/
                             (_dataNorm[i] * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res,_l-1);
}
//here
void zeroOneRBiasKernel::dot(vec &v,Ref<VectorXd> res){
  VectorXd tmp(_p);
  double normVec = 0.0;
  for(size_t i=0; i<_p;++i){
    tmp(i) = v[i];
    normVec += v[i]*v[i];
  }
  normVec = sqrt(normVec + _bias[0]);
  ArrayXd alpha = OneDpi*(((((_data.transpose()*tmp).array()+_bias[0])/
                            (normVec * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res,_l-1);
}
void zeroOneRBiasKernel::dot(const Ref<const  VectorXd> &v,Ref<VectorXd> res){
  ArrayXd alpha = OneDpi*(((((_data.transpose()*v).array()+_bias[0])/
                            (sqrt(v.squaredNorm()+_bias[0]) * _dataNorm)).min(1)).max(-1)).acos();
  calc(alpha,res,_l-1);
}
size_t zeroOneRBiasKernel::getN(){
  return _n;
}
string zeroOneRBiasKernel::getName(){
  string st = "ZeroOneRBias\t"+to_string(_l);
  for(size_t l=0;l<_l;++l){
    st += "\t"+to_string(_hidden[l]);
  } 
  for(size_t l=0;l<_l;++l){
    st += "\t"+to_string(_bias[l]);
  } 
  
  return st;
}
