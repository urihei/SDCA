#include "learnMultiClassSVM.hpp"
#include "usedFun.hpp"

learnMultiClassSVM::learnMultiClassSVM(ivec &y,matd &data,size_t k,
                                       unsigned int iter,unsigned int accIter,
                                       double lambda)
  :learnSVM(y,data,iter,accIter,lambda),_k(k){
  _gamma = 0.1;
}
void learnMultiClassSVM::buildTempVariables(){
  _prm = new unsigned int[_n];

  _p.resize(_k);

  _mu.resize(_k);
  _a.resize(_k);

  _muh.resize(_k);
  _mub.resize(_k);
  _z.resize(_k);

  _OneToK.resize(_k);
  
  _OneToK.setOnes();
  cumsum(_OneToK.data(),_k,_OneToK.data());

  _normOne.resize(_k);

  _squaredNormData.resize(_n);
  _squaredNormData = _data.diagonal();
  _data.diagonal().setZero();

  _zALPHAtimeK.resize(_k,_n);
}
void learnMultiClassSVM::unBuildTempVariables(){
  delete _prm;
  _data += _squaredNormData.asDiagonal();
}
void learnMultiClassSVM::learn_SDCA(mat &alpha, mat &zALPHA){
  buildTempVariables();
  learn_SDCA_Iterations(alpha, zALPHA);
  unBuildTempVariables();
}
void learnMultiClassSVM::learn_SDCA_Iterations(mat &alpha, mat &zALPHA){


  double lambdaN = 1/(_n*_lambda);

  double gammaLambdan = _gamma*_n*_lambda;


  double C;
  unsigned int ind = 0;
  
  _zALPHAtimeK = zALPHA *_data+zALPHA_squaredNormData.asDiagonal();

  for(unsigned int t=0;t<_iter;++t){
    if((ind % _n) == 0){
      randperm(_n,_prm);
      ind = 0;
    }
    size_t i = _prm[ind];
    size_t curLabel = _y[i];
    //
    // AlTemp = alpha;
    // AlTemp.col(i).setZero();
    // AlTemp = AlTemp + zALPHA;

    // p = lambdaN * (AlTemp * _data.col(i));
    // p = p - p(curLabel);

    // c.setOnes();
    // c(curLabel) = 0;
    //
    _p = lambdaN * (alpha * _data.col(i) + _zALPHAtimeK.col(i));
    _p = _p - _p(curLabel) +1;
    //        cerr<<"---pt---\n"<<endl<<pt-1<<endl<<"---p---\n"<<p<<endl;
    _p(curLabel) = 0;
    // cerr<<"DIS:"<<(pt-p-c).matrix().squaredNorm() <<endl;
    //
    _mu = _p/(_gamma+(_squaredNormData(i)*lambdaN));
    C = 1/(1+(gammaLambdan/_squaredNormData(i)));

    //optimizeDual_SDCA(mu,C,a);

    //creating muh
    _muh = _mu.max(0);
    sort(_muh.data(),_muh.data()+_k);
    _muh.reverseInPlace();
    //creating mub
    cumsum(_muh.data(),_k,_mub.data());
    //creating z
    _z  = (_mub - (_OneToK * _muh)).min(1);
    //calc normOne (matlab mubDjC)
    _normOne = _mub/(1+(_OneToK*C));
    //find indF
    //size_t indF = findFirstBetween(normOne.data(),z.data(),_k);
    size_t indF = 0;
    size_t indJ = _k+1;
        
    while((indF<_k-1)&&((_normOne(indF) < _z(indF)) || (_normOne(indF) > _z(indF+1)))){
      if(indJ == _k+1 && _z(indF) ==1){
	indJ = indF -1;
      }
      indF++;
    }
    if((indF==_k-1)&& !((_normOne(indF)>=_z(indF))&& (_normOne(indF) <= 1)) ){
      if(indJ == _k+1 && _z(indF) ==1){
	indJ = indF -1;
      }
      indF++;
    }

    if(indF >= _k){
      //            size_t indJ = findFirst(z.data(),_k)-1;
      _a = (_mu+((1-_mub(indJ))/(indJ+1))).max(0);
      if(((_a-_mu).matrix().squaredNorm()+C) > (_mu.matrix().squaredNorm())){
	_a.setZero();
      }
    }else{
      _a = (_mu+((_normOne(indF)-_mub(indF))/(indF+1))).max(0);
    }
    // END optimizeDual_SDCA
    alpha.col(i) = -_a;
    alpha(curLabel,i) = _a.matrix().lpNorm<1>();
                
    ind++;
  }
}
void learnMultiClassSVM::acc_learn_SDCA(mat &alpha){
  double kappa = 100*_lambda;
  double mu = _lambda/2;
  double rho = mu+kappa;
  double eta = sqrt(mu/rho);
  double beta = (1-eta)/(1+eta);

  MatrixXd zALPHA(_k,_n);
  zALPHA.setZero();
  MatrixXd ALPHA_t(_k,_n);
  ALPHA_t.setZero();
  MatrixXd zALPHA_t(_k,_n);

  buildTempVariables();

  for(unsigned int t =0;t<_accIter;++t){
    learn_SDCA_Iterations(alpha, zALPHA);
    zALPHA_t = zALPHA;
    zALPHA = (1+beta)*(zALPHA + alpha) - beta * ALPHA_t;
    ALPHA_t = zALPHA_t+alpha;
  }
  unBuildTempVariables();  
  alpha = ALPHA_t;
}
void learnMultiClassSVM::returnModel(mat &model){

}
void learnMultiClassSVM::setGamma(double gamma){
  _gamma = gamma;
}
double learnMultiClassSVM::getGamma(){
  return _gamma;
}
