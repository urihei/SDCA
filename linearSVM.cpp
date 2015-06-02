#include "linearSVM.hpp"
#include "usedFun.hpp"

linearSVM::linearSVM(size_t* y,double* data,size_t k,size_t n,size_t p,
                     double lambda, double gamma,
                     unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter),_data(data,p,n){
  _n = n;
  _usedN = _n;
  _p = p;
  _y = y;
  _W.resize(_p,_k);
  _squaredNormData.resize(_n);
  _prmArray.resize(_n);
  for(size_t i = 0;i<_n;++i){
    _squaredNormData(i) = _data.col(i).squaredNorm();
    _prmArray[i] = i;
  }
}
double linearSVM::learn_SDCA(){
  MatrixXd alpha(_k,_n);
  alpha.setZero();
  MatrixXd zW(_p,_k);
  zW.setZero();
  return learn_SDCA(alpha, zW,_eps);
}

double linearSVM::learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zW){
  return learn_SDCA(alpha, zW,_eps);
}
double linearSVM::learn_SDCA(Ref<MatrixXd> alpha, const Ref<const MatrixXd> &zW,double eps){


  double lambdaN = 1/(_usedN*_lambda);

  double gammaLambdan = _gamma*_usedN*_lambda;


  ivec prm(_usedN);
  

  unsigned int ind = 0;

  ArrayXd p(_k);
  ArrayXd mu(_k);
  //ArrayXd a(_k);
  double C;
  

  _W = zW + lambdaN * _data *alpha.transpose(); 

  double gap = eps + 1;
    
  for(unsigned int t=1;t<=_iter;++t){
    if((ind % _usedN) == 0){
      randperm(_usedN,prm,_prmArray);
      ind = 0;
      // cerr<<endl;
      // for(size_t i=0; i<_usedN;++i){
      //   cerr<<prm[i]<<" ";
      // }
      // cerr<<endl;
    }
        
    size_t i = prm[ind];
    size_t curLabel = _y[i];

    _W -= lambdaN * _data.col(i)*alpha.col(i).transpose();
    
    p = _W.transpose() * _data.col(i);
    p -= p(curLabel) -1;
    p(curLabel) = 0;



    mu = p/(_gamma+(_squaredNormData(i)*lambdaN));
    C = 1/(1+(gammaLambdan/_squaredNormData(i)));



    //    optimizeDual_SDCA(mu,C,a);
    optimizeDual_SDCA(mu,C,alpha,i,curLabel);
    // END optimizeDual_SDCA

    //alpha.col(i)     = - a;
    //alpha(curLabel,i) = a.matrix().lpNorm<1>();

        
    _W += lambdaN * _data.col(i)*alpha.col(i).transpose(); 
    if( t % ( _usedN* _checkGap) == 0){//
      gap = getGap(alpha,zW);
            
    }
        
    if(gap < eps)
      break;

    ind++;
  }
  if(gap >eps){
    gap = getGap(alpha,zW);
  }
  return gap;
}

double linearSVM::learn_acc_SDCA(){
  double kappa = 100*_lambda;
  double mu = _lambda/2;
  double rho = mu+kappa;
  double eta = sqrt(mu/rho);
  double beta = (1.0-eta)/(1.0+eta);

  mat alpha(_k,_n);
  alpha.setZero();

  mat zW(_p,_k);
  zW.setZero();
    
  mat W_t(_p,_k);
  W_t.setZero();

    
  double gap = _eps + 1.0;
  double epsilon_t;


  double OnePetaSquare = 1+1/eta*1/eta;
  double xi = OnePetaSquare * (1-_gamma/(2*(_k-1)));
  eta = eta /2;
    
    
  for(unsigned int t =1; t<=_accIter; ++t){
    epsilon_t = learn_SDCA(alpha, zW,eta/OnePetaSquare * xi);
    if(t%_checkGapAcc ==0){
      if(_verbose)
        cerr<<"ACC iter: "<<t<<" gap: ";
      gap = (1+rho/mu)*epsilon_t + 
        (rho*kappa)/(2*mu)*(_W-zW).squaredNorm();
      if(_verbose)
        cerr<<gap<<endl;
    }
    if(gap < _eps)
      break;
    zW = (1+beta)*_W - beta * W_t;
    W_t = _W*1;
    xi = xi * (1-eta);
  }
  return gap;
}

double linearSVM::getGap(const Ref<const MatrixXd> &alpha, const Ref<const MatrixXd> &zW){
  double pr = 0.0;
  double du = 0.0;

  ArrayXd a(_k);
  ArrayXd b(_k);
    
  for(size_t ii = 0; ii<_usedN;++ii){
    size_t i= _prmArray[ii];
    size_t currentLabel = _y[i];
      
    a = _W.transpose() * _data.col(i);
    a = (a - a(currentLabel) +1)/_gamma;
    a(currentLabel) = 0;
      
    project_SDCA(a,b);
      
    b = b - a;
    pr += _gamma/2 * (a.matrix().squaredNorm() - b.matrix().squaredNorm());
    du -= alpha.col(i).sum() - alpha(currentLabel,i) +
      _gamma/2 * (alpha.col(i).squaredNorm() - alpha(currentLabel,i)*alpha(currentLabel,i));
        
  }
  pr = pr/_usedN + _lambda * (_W.array() * (_W.array() - zW.array())).sum();
  du /= _usedN;
  double gap = pr - du;
  if(_verbose)
    cerr<< "primal: "<<pr<<"\t dual: "<<du<<"\t Gap: "<<gap<<endl;
  return gap;
}
double linearSVM::getGap(){
  mat zW(_p,_k);
  zW.setZero();
  mat alpha(_k,_usedN);
  alpha.setZero();
  return getGap(alpha,zW);
}
void linearSVM::classify(double* data,size_t* res,size_t n_test,size_t p_test){
  Map<Matrix<double,Dynamic,Dynamic,ColMajor>> mData(data,p_test,n_test);
  //  cerr<<mData<<endl;
  classify(mData,res);
}
void linearSVM::classify(matd &data,size_t* res){
  MatrixXd mData;
  fillMatrix(data,mData);
  //    cerr<<"Data rows: "<<mData.rows()<<" cols: "<<mData.cols()<<endl;
  mData.transposeInPlace();
  classify(mData,res);
}
void linearSVM::classify(const Ref<const MatrixXd> &mData,size_t* res){
  size_t n = mData.cols();

  MatrixXd ya(_k,n);
  ya = _W.transpose()* mData;
  MatrixXf::Index index;
  for(size_t i=0;i<n;i++){
    ya.col(i).maxCoeff(&index);
    res[i] = (size_t) index;
    // cerr<<ya.col(i)<<endl;
    // cerr<<res[i]<<endl;
  }
}
void linearSVM::classify(ivec_iter& itb ,ivec_iter& ite,ivec &res){
  size_t n = std::distance(itb,ite);
  if(res.size() != n){
    res.resize(n);
  }
  MatrixXf::Index index;
  size_t i=0;
  for(ivec_iter it =itb; it<ite;++it){
    (_W.transpose() * (_data.col(*it))).array().maxCoeff(&index);
    res[i++] = (size_t) index;   
  }
} 

void linearSVM::saveModel(FILE* pFile){
  saveModel(pFile,"Linear",_W);
}
void linearSVM::setParameter(matd &par){
  fillMatrix(par,_W);
}

void linearSVM::init(){
  _W.setZero();
}
