#include "baseKernelSVM.hpp"
#include "usedFun.hpp"

baseKernelSVM::baseKernelSVM(size_t k, double lambda, double gamma , unsigned int iter, unsigned int accIter):svm(k,lambda,gamma,iter,accIter){}

double baseKernelSVM::getGap(const Ref <const MatrixXd> &alpha,const Ref <const MatrixXd> &zALPHA){
  double pr = 0.0;
  double du = 0.0;

  double lambdaN  = 1/(_lambda * _usedN);

  mat az = zALPHA + alpha;
  ArrayXd a(_k);


  double normPart = 0.0;

  VectorXd kerCol(_n);    
  ArrayXd b(_k);
    
  for(size_t ii = 0; ii<_usedN;++ii){
    size_t i = _prmArray[ii];
    size_t currentLabel = _y[i];
      
    getCol(i,kerCol);//_ker->dot(i,kerCol);
    a = lambdaN * az * kerCol;

    normPart += lambdaN*(a * alpha.col(i).array()).sum();
        
    a = (a - a(currentLabel)  + 1)/_gamma;
    a(currentLabel) = 0;
    project_SDCA(a,b);
    b = b - a;
    pr += _gamma/2 * (a.matrix().squaredNorm() - b.matrix().squaredNorm());
    du -= alpha.col(i).sum() - alpha(currentLabel,i) +
      _gamma/2 * (alpha.col(i).squaredNorm() - alpha(currentLabel,i)*alpha(currentLabel,i));
  }
  pr = pr/_usedN+_lambda*normPart;
  du /= _usedN;
  double gap = pr - du;
  if(_verbose)
    fprintf(stderr,"primal %g\t dual %g\t Gap %g \n",pr,du,gap);
  return gap;
}
double baseKernelSVM::getGap(){
  mat zAlpha(_k,_n);
  zAlpha.setZero();
  return getGap(_alpha,zAlpha);
}

double baseKernelSVM::learn_SDCA(){
  mat zALPHA(_k,_n);
  zALPHA.setZero();
  return learn_SDCA(_alpha,zALPHA,_eps);
}

double baseKernelSVM::learn_SDCA(Ref <MatrixXd> alpha, const Ref <const MatrixXd> &zALPHA){
  return learn_SDCA(alpha,zALPHA,_eps);
}


double baseKernelSVM::learn_acc_SDCA(){
  double kappa = 10/_usedN;//100*_lambda;
  double mu = _lambda/2;
  double rho = mu+kappa;
  double eta = sqrt(mu/rho);
  double beta = (1-eta)/(1+eta);

  mat alpha(_k,_n);
  alpha.setZero();
  mat zALPHA(_k,_n);
  zALPHA.setZero();
  MatrixXd zALPHA_t(_k,_n);

  double gap = _eps + 1.0;
  VectorXd kerCol(_n);
  double epsilon_t;

  double OnePetaSquare = 1+1/eta*1/eta;
  double xi = OnePetaSquare * (1-_gamma/(2*(_k-1)));
  eta = eta /2;

    
  for(unsigned int t =1;t<=_accIter;++t){

    epsilon_t = learn_SDCA(alpha,zALPHA,eta/OnePetaSquare * xi);
    zALPHA_t = zALPHA;
    zALPHA = (1+beta)*(zALPHA + alpha) - beta * _alpha;
    _alpha = zALPHA_t+alpha;
    if(t%_checkGapAcc ==0){
      if(_verbose)
	cerr<<"ACC iter: "<<t<<" gap: ";
      double diff = 0;
      // sum(diag(alpha * K * alpha'))
      for(size_t nn=0;nn<_usedN;++nn){
	size_t n = _prmArray[nn];
	getCol(n,kerCol);
	for(size_t k=0; k<_k;++k){
	  diff += alpha(k,n)*kerCol.dot(alpha.row(k));
	}
      }
      gap = (1+rho/mu)*epsilon_t + (rho*kappa)/(2*mu)*diff;
      if(_verbose)
	cerr<<gap<<endl;
    }
    if(gap < _eps)
      break;
    xi = xi * (1-eta);
  }
  return gap;
}

void baseKernelSVM::setParameter(matd &par){
  fillMatrix(par,_alpha);
}

void baseKernelSVM::init(){
  _alpha.setZero();
}
