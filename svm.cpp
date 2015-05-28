#include "svm.hpp"
#include "usedFun.hpp"

svm::svm(size_t k,double lambda, double gamma,unsigned int iter, unsigned int accIter):_iter(iter),_accIter(accIter),_lambda(lambda),_gamma(gamma),_k(k){
  _OneToK.resize(_k);
  _OneToK.setOnes();
  cumsum(_OneToK.data(),_k,_OneToK.data());
  _eps = 1e-3;
  _checkGap = 5;
  _checkGapAcc = 5;
  _verbose = true;
  initRand();
}
svm::~svm(){};
void svm::setIter(unsigned int iter){
  _iter = iter;
}
void svm::setAccIter(unsigned int iter){
  _accIter = iter;
}
void svm::setCheckGap(unsigned int checkGap){
  _checkGap = checkGap;
}
void svm::setCheckGapAcc(unsigned int check){
  _checkGapAcc = check;
}
void svm::setLambda(double lambda){
  _lambda = lambda;
}
void svm::setGamma(double gamma){
  _gamma = gamma;
}
void svm::setEpsilon(double epsilon){
  _eps = epsilon;
}
void svm::setVerbose(bool ver){
  _verbose = ver;
}

void svm::setUsedN(size_t n){
  _usedN = n;
}

void svm::samplePrm(){
  ivec tmp(_n);
  randperm(_n,tmp,_prmArray);
  _prmArray = tmp;
}

ivec_iter svm::getPrmArrayBegin(){
  return _prmArray.begin();
}
ivec_iter svm::getPrmArrayEnd(){
  return _prmArray.end();
}

void svm::shiftPrm(size_t n){
  std::rotate(_prmArray.begin(),_prmArray.begin()+n,_prmArray.end());
}
size_t svm::getN(){
  return _n;
}
unsigned int svm::getIter(){
  return _iter;
}
unsigned int svm::getAccIter(){
  return _accIter;
}
unsigned int svm::getCheckGap(){
  return _checkGap;
}
unsigned int svm::getCheckGapAcc(){
  return _checkGapAcc;
}
double svm::getLambda(){
  return _lambda;
}
double svm::getGamma(){
  return _gamma;
}
double svm::getEpsilon(){
  return _eps;
}
bool svm::getVerbose(){
  return _verbose;
}
double svm::optimizeDual_SDCA(vec &mu,double C,vec &a){
  vec muh(_k);
  vec mub(_k);
  vec z(_k);
  vec normOne(_k);
  for(size_t k=0;k<_k;++k){
    muh[k] = (mu[k] > 0)? mu[k]:0;
  }
  sort(muh.begin(),muh.end(),std::greater<double>());
  //cumsum,z
  mub[0] = muh[0];
  z[0] = 0;
  normOne[0] = mub[0]/(1+C);
  for(size_t k=1;k<_k;++k){
    mub[k] = mub[k-1]+muh[k];
    z[k] = mub[k] - (k+1)*muh[k];
    z[k] = (z[k]>1)? 1:z[k];
    normOne[k] = mub[k]/(1+(k+1)*C);
  }
  //
  size_t indF = 0;
  size_t indJ = _k+1;
  while((indF<_k-1) &&((normOne[indF] < z[indF]) || (normOne[indF] > z[indF+1]))){
    //finding the J index (for the case there is no index F)
    if(indJ == _k+1 && z[indF] ==1){
      indJ = indF -1;
    }
    indF++;
  }
  if((indF==_k-1)&& !((normOne[indF]>=z[indF])&& (normOne[indF] <= 1)) ){
    if(indJ == _k+1 && z[indF] ==1){
      indJ = indF -1;
    }
    indF++;
  }
  //  cerr<<"Before if: indF "<< indF<<" indJ "<<indJ<<endl;
    
  if(indF >= _k){
    double divConst = (1-mub[indJ])/(indJ+1);
    double aMmSn = 0;
    double mSn = 0;
    double tmp;
    for(size_t k=0; k<_k;++k){
      a[k] = mu[k]+divConst;
      a[k] = (a[k] < 0)? 0:a[k];
      tmp = a[k]-mu[k];
      aMmSn += tmp*tmp;
      mSn += mu[k]*mu[k];
    }
    if(aMmSn+C > mSn){
      for(size_t k=0; k<_k;++k){
        a[k] =0;
      }
      return 0;
    }
    return 1;
  }
  double divConst = (normOne[indF]-mub[indF])/(indF+1);
  double norm1A = 0;
  for(size_t k=0; k<_k;++k){
    a[k] = mu[k]+ divConst;
    a[k] = (a[k]<0)? 0:a[k];
    norm1A += a[k];
  }
  return norm1A;
}
void svm::optimizeDual_SDCA(const Ref<const ArrayXd> &mu,double C,Ref<MatrixXd> a,size_t i,size_t yi){
  //void svm::optimizeDual_SDCA(ArrayXd &mu,double C,ArrayXd &a){
  ArrayXd muh(_k);
  ArrayXd mub(_k);
  ArrayXd z(_k);   
  ArrayXd normOne(_k);

  //creating muh
  muh = mu.max(0);
  sort(muh.data(),muh.data()+_k,std::greater<double>());
  //muh.reverseInPlace();
  //creating mub
  cumsum(muh.data(),_k,mub.data());
  //creating z
  z  = (mub - (_OneToK * muh)).min(1);
  //calc normOne (matlab mubDjC)
  normOne = mub/(1+(_OneToK*C));
  //find indF
  //        size_t indF = findFirstBetween(normOne.data(),z.data(),_k);
  size_t indF = 0;
  size_t indJ = _k+1;
        
  while((indF<_k-1)&&((normOne(indF) < z(indF)) || (normOne(indF) > z(indF+1)))){
    if(indJ == _k+1 && z(indF) ==1){
      indJ = indF -1;
    }
    indF++;
  }
  if((indF==_k-1)&& !((normOne(indF)>=z(indF))&& (normOne(indF) <= 1)) ){
    if(indJ == _k+1 && z(indF) ==1){
      indJ = indF -1;
    }
    indF++;
  }

  if(indF >= _k){
    // size_t indJ = findFirst(z.data(),_k)-1;
    a.col(i) = (mu+((1-mub(indJ))/(indJ+1))).max(0);
    //a  = (mu+((1-mub(indJ))/(indJ+1))).max(0);
    if(((a.col(i).array() - mu).matrix().squaredNorm()+C) > (mu.matrix().squaredNorm())){
      //if(((a - mu).matrix().squaredNorm()+C) > (mu.matrix().squaredNorm())){
      a.col(i).setZero();
      //a.setZero();
    }else{
      a.col(i) = -1* a.col(i);
      a(yi,i) = 1;
    }
  }else{
    a.col(i) = -1*(mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
    //a = (mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
    a(yi,i) = normOne(indF);
  }
}
//retrun normsquare of a-b and b = b-a 
double svm::project_SDCA(vec &mu,vec &b){
  vec muh(_k);
  vec mub(_k);
  vec z(_k);   
  //
  double norm1muh = 0;
  for(size_t k=0;k<_k;++k){
    muh[k] = (mu[k] > 0)? mu[k]:0;
    norm1muh += muh[k];
  }
  
  if(norm1muh <= 1){
    double normB = 0;
    for(size_t k=0;k<_k;++k){
      b[k] = muh[k]-mu[k];
      normB += b[k]*b[k];
    } 
    return normB;
  }
  
  sort(muh.begin(),muh.end(),std::greater<double>());
  mub[0] = muh[0];
  z[0] = 1;
  for(size_t k=1;k<_k;++k){
      mub[k] = mub[k-1]+muh[k];
      z[k] = 1+(k+1)*muh[k]-mub[k];
  }
  size_t ind = _k-1;
  while( ind > 0 && z[ind]<= 0){
    ind --;
  }

  double constM = (1-mub[ind])/(ind+1);
  double normB = 0;
  for(size_t k=0;k<_k;k++){
    b[k] = mu[k] + constM;
    b[k] = (b[k] < 0)? 0:b[k];
    b[k] = b[k]-mu[k];
    normB += b[k]*b[k];
  }
  return normB;
}
void svm::project_SDCA(const Ref<const ArrayXd> &mu,Ref<ArrayXd> b){
  ArrayXd muh(_k);
  ArrayXd mub(_k);
  ArrayXd z(_k);   
  //
  muh = mu.max(0);
  if(muh.sum() <= 1){
    b = muh;
    return;
  }
  sort(muh.data(),muh.data()+_k,std::greater<double>());
  
  cumsum(muh.data(),_k,mub.data());

  z = 1 + _OneToK * muh - mub;

  size_t ind = _k-1;
  while( ind > 0 && z(ind)<= 0){
    ind --;
  }
  b = mu+ (1-mub(ind))/(ind+1);
  b = b.max(0);
}

void svm::saveModel(FILE* pFile, string kernel,const Ref<const MatrixXd> &model){
  fprintf(pFile,"%s\t%g\t%g\n",kernel.c_str(),_lambda,_gamma);
  for(int i=0;i<model.rows();++i){
    for(int j=0; j<model.cols();++j){
      fprintf(pFile,"%g ",model(i,j));
    }
    fprintf(pFile,"\n");
  }
    
}
void svm::saveModel(FILE* pFile, string kernel,const matd &model){
  fprintf(pFile,"%s\t%g\t%g\n",kernel.c_str(),_lambda,_gamma);
  for(size_t i=0;i<model.size();++i){
    for(size_t j=0; j<model[i].size();++j){
      fprintf(pFile,"%g ",model[i][j]);
    }
    fprintf(pFile,"\n");
  }
    
}
