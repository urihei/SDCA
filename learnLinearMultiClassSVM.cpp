#include "learnLinearMultiClassSVM.hpp"
#include "usedFun.hpp"

learnLinearMultiClassSVM::learnLinearMultiClassSVM(ivec &y,matd &data,size_t k,
                                       unsigned int iter,unsigned int accIter,
                                       double lambda)
  :learnSVM(y,data,iter,accIter,lambda),_k(k){
  _gamma = 0.1;
  _sRow = _data.rows();
}
void learnLinearMultiClassSVM::buildTempVariables(){
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

}
void learnLinearMultiClassSVM::unBuildTempVariables(){
  delete _prm;
}
void learnLinearMultiClassSVM::learn_SDCA(mat &alpha, mat &zW){
  buildTempVariables();
  learn_SDCA_Iterations(alpha, zW);
  unBuildTempVariables();
}
void learnLinearMultiClassSVM::learn_SDCA_Iterations(mat &alpha, mat &W){


  double lambdaN = 1/(_n*_lambda);

  double gammaLambdan = _gamma*_n*_lambda;


  double C;
  unsigned int ind = 0;
  
  W += lambdaN * _data *alpha.transpose(); 

  for(unsigned int t=0;t<_iter;++t){
    if((ind % _n) == 0){
      randperm(_n,_prm);
      ind = 0;
    }
    size_t i = _prm[ind];
    size_t curLabel = _y[i];
    W -= lambdaN * _data.col(i)*alpha.col(i).transpose();
    _p = W.transpose() * _data.col(i);
    _p -=  _p(curLabel) +1;
    _p(curLabel) = 0;

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
    W += lambdaN * _data.col(i)*alpha.col(i).transpose(); 
                
    ind++;
  }
}
void learnLinearMultiClassSVM::acc_learn_SDCA(mat &zW){
  double kappa = 100*_lambda;
  double mu = _lambda/2;
  double rho = mu+kappa;
  double eta = sqrt(mu/rho);
  double beta = (1-eta)/(1+eta);

  MatrixXd alpha(_k,_n);
  alpha.setZero();
  MatrixXd W_t(_sRow,_k);
  MatrixXd W_tt(_sRow,_k);
  W_t.setZero();
  buildTempVariables();

  for(unsigned int t =0;t<_accIter;++t){
    learn_SDCA_Iterations(alpha, zW);
    W_tt = zW;
    zW = (1+beta)*zW - beta * W_t;
    W_t = W_tt;
  }
  unBuildTempVariables();  
  zW = W_t;
}
void learnLinearMultiClassSVM::returnModel(mat &model){

}
void learnLinearMultiClassSVM::setGamma(double gamma){
  _gamma = gamma;
}
double learnLinearMultiClassSVM::getGamma(){
  return _gamma;
}
