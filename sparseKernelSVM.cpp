#include "sparseKernelSVM.hpp"
#include "usedFun.hpp"

sparseKernelSVM::sparseKernelSVM(ivec &y,Kernel* ker, size_t k,
                                 double lambda, double gamma,
                                 unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter),_alpha(sparseAlpha(k,y.size())){
  _y = y;//reassign;
  _ker = ker;
  _n = _ker->getN();
  _usedN = _n;
  _alpha.setK(_k);
  _alpha.setP(_n);
  _squaredNormData.resize(_n);
  _prmArray.resize(_n);
  for(size_t i = 0;i<_n;++i){
    _squaredNormData(i) = _ker->squaredNorm(i);
    _prmArray[i] = i;
  }
}
double sparseKernelSVM::learn_SDCA(){
  matd pOld(_k,vec(_n,0));
  
  return learn_SDCA(_alpha,pOld,_eps);
}
double sparseKernelSVM::learn_SDCA(sparseAlpha &alpha,matd &pOld,double eps){
  double lambdaN = 1/(_usedN*_lambda);
    
  double gammaLambdan = _gamma*_usedN*_lambda;


  double C;
  unsigned int ind = 0;

  ivec prm(_usedN);
  vec kerCol(_usedN);
    
    
  vec p(_k);
  vec mu(_k);
  vec a(_k);

  vector<map<size_t,double>::iterator> indx(_k);
  double gap = eps + 1;
  unsigned int t=1;
  for(;t <= _iter;++t){
    if((ind % _usedN) == 0){
      randperm(_usedN,prm,_prmArray);
      ind = 0;
    }
    size_t i = prm[ind];
    size_t curLabel = _y[i];

    //        _ker->dot(i,kerCol);
    //        alpha.col(i).setZero();
    // p = lambdaN * ((alpha+zALPHA) * kerCol);

    alpha.vecMul(p,lambdaN,_ker,i,indx);
    //p += pOld(:,i)
          
    //p -= p(curLabel) - 1;
    //        mu = p/(_gamma+(_squaredNormData(i)*lambdaN));
    double pIcuLabel = p[curLabel]  + pOld[curLabel][i];
    double normConst = (_gamma+(_squaredNormData(i)*lambdaN));
    for(size_t k= 0;k<_k;++k){
      mu[k] = (p[k] + pOld[k][i] - pIcuLabel +1.0)/normConst;
    }
    mu[curLabel] = 0;
        
    //p[curLabel] = 0;

        
    C = 1/(1+(gammaLambdan/_squaredNormData(i)));

    //note alpha is changing here
    double norm1A = optimizeDual_SDCA(mu,C,a);

    for(size_t k=0;k<_k;++k){
      double val = (k == curLabel)? norm1A:-a[k];
      if(val != 0){
        if(indx[k] != alpha.getEnd(k)){
          indx[k]->second = val;
        }else{
          alpha.insert(k,i,val);
        }
      }
            
    }
    if(t % (_usedN* _checkGap) == 0){
      gap = getGap(alpha,pOld);
            
    }
        
    if(gap < eps)
      break;
        
    ind++;
  }

  if(gap >eps && (t % (_usedN* _checkGap) != 0)){
    gap = getGap(alpha,pOld);
  }
  return gap;
}

double sparseKernelSVM::learn_acc_SDCA(){
  double kappa = 10/_usedN;//100*_lambda;
  double mu = _lambda/2;
  double rho = mu+kappa;
  double eta = sqrt(mu/rho);
  double beta = (1-eta)/(1+eta);

  //mat alpha(_k,_n);
  //  alpha.setZero();
  sparseAlpha alpha(_k,_usedN);
  
  sparseAlpha Zalpha(_k,_usedN);
  
  matd pOld(_k,vec(_usedN,0));

  //  MatrixXd zALPHA_t(_k,_n);
  matd pOld_t(_k,vec(_usedN,0));
  
  double gap = _eps + 1.0;

  //VectorXd kerCol(_n);
  vec kerKol(_usedN);
  
  double epsilon_t;

  double OnePetaSquare = 1+1/eta*1/eta;
  double xi = OnePetaSquare * (1-_gamma/(2*(_k-1)));
  eta = eta /2;

  vector<map<size_t,double>::iterator> empty(_k);
  vec resSample(_k);
  for(unsigned int t =1;t<=_accIter;++t){
    epsilon_t = learn_SDCA(alpha,pOld,eta/OnePetaSquare * xi);
    //wrong need to fix.
    _alpha.add(alpha);
    for(size_t n=0;n<_usedN;++n){
      size_t col = _prmArray[n];
      _alpha.vecMul(resSample,1/(_lambda*_usedN),_ker,col,empty,true);
      for(size_t k=0; k<_k;++k){
        pOld[k][col] = (1+beta)*resSample[k] - beta*pOld[k][col];
      }
    }
    
    //zALPHA_t = zALPHA;
    //    zALPHA = (1+beta)*(zALPHA + alpha) - beta * _alpha;
    //_alpha = zALPHA_t+alpha;
    if(t%_checkGapAcc ==0){
      if(_verbose)
	cerr<<"ACC iter: "<<t<<" gap: ";
      // double diff = 0;
      // // sum(diag(alpha * K * alpha'))
      // for(size_t nn=0;nn<_usedN;++nn){
      //   size_t n = _prmArray[nn];
      //   getCol(n,kerCol);
      //   for(size_t k=0; k<_k;++k){
      //     diff += alpha(k,n)*kerCol.dot(alpha.row(k));
      //   }
      // }
      double diff = alpha.norm(_ker);
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
double sparseKernelSVM::getGap(){
  matd tmp(_k,vec(_n,0));
  return getGap(_alpha,tmp);
}
double sparseKernelSVM::getGap(sparseAlpha &alpha,matd &pOld){
  double pr = 0.0;
  double du = 0.0;

  double lambdaN  = 1/(_lambda * _usedN);

  vec a(_k);


  double normPart = 0.0;

  vec kerCol(_usedN);    
  vec b(_k);
  vector<map<size_t,double>::iterator> indx(_k);

  for(size_t ii = 0; ii<_usedN;++ii){
    size_t i = _prmArray[ii];
    size_t currentLabel = _y[i];
      
    // getCol(i,kerCol);//_ker->dot(i,kerCol);
    // a = lambdaN * az * kerCol;
    alpha.vecMul(a,lambdaN,_ker,i,indx,true);
    double sumTmp = 0; //(a * alpha.col(i).array()).sum()
    double normSa = 0; //|a|^2 
    double norm1alpha = 0; //|alpha|_1 - c'*alpha
    double normSalpha = 0; //|alpha|^2
    double aCurrent = a[currentLabel]+pOld[currentLabel][i];
    for(size_t k=0;k<_k;++k){
      a[k] += pOld[k][i];
      if(indx[k] != alpha.getEnd(k)){
        sumTmp += a[k] * indx[k]->second; // normPart += lambdaN*(a * alpha.col(i).array()).sum();
        norm1alpha += indx[k]->second;
        normSalpha += indx[k]->second*indx[k]->second;
      }
      a[k] = (a[k]-aCurrent+1)/_gamma;      //    a = (a - a(currentLabel)  + 1)/_gamma;
      normSa += a[k]*a[k];
      
    }
    normSa -= (1.0/_gamma) *(1.0/_gamma); // the value of a[currentLabel] needed to be removed.

    double alphaAtLabel = 0;
    if(indx[currentLabel] != alpha.getEnd(currentLabel)){
      alphaAtLabel = indx[currentLabel]->second;
    }
    
    norm1alpha -= alphaAtLabel;
    a[currentLabel] = 0;
     
    
    normPart += lambdaN*sumTmp;

    //b = b - a;
    double normBs = project_SDCA(a,b);
    pr += _gamma/2 * (normSa - normBs);
    du -= norm1alpha + _gamma/2 *(normSalpha - alphaAtLabel* alphaAtLabel );
  }
  pr = pr/_usedN+_lambda*normPart;
  du /= _usedN;
  double gap = pr - du;
  if(_verbose)
    fprintf(stderr,"primal %g\t dual %g\t Gap %g \n",pr,du,gap);
  return gap;
}


void sparseKernelSVM::classify(matd &data, ivec &res){
  size_t n = data.size();
  if(res.size() != n){
    res.resize(n);
  }
  vec ya(_k);
  //  vec kerCol(_n);
  cerr<<"Start kernel classify"<<endl;
  for(size_t i=0;i<n;++i){
    //_ker->dot(data[i],kerCol);
    // ya = _alpha * kerCol;
    // ya.maxCoeff(&index);
    //  res[i] = (size_t) index;
    res[i] = _alpha.vecMul(ya,1/_lambda,_ker,data[i]); //build a new function that recive vector and not column.
  }
    
}
void sparseKernelSVM::classify(ivec_iter &itb,ivec_iter &ite,ivec &res){
  size_t n = std::distance(itb,ite);
  if(res.size() != n){
    res.resize(n);
  }
  //  vec kerCol(_n);
  vector<map<size_t,double>::iterator> empty;
  vec ya(_k);
  size_t i =0;
  for(ivec_iter it =itb; it<ite;++it){
    //_ker->dot(*it,kerCol);
    //(_alpha * kerCol).array().maxCoeff(&index);
    //res[i++] = (size_t) index;
    res[i++] = _alpha.vecMul(ya,1/_lambda,_ker,*it,empty);
  }
}

void sparseKernelSVM::saveModel(FILE* pFile){
  matd empty;
  saveModel(pFile,_ker->getName(),empty);
  _alpha.write(pFile);
}
void sparseKernelSVM::getCol(size_t i,Ref <VectorXd>  kerCol){
  _ker->dot(i,kerCol);
}
void sparseKernelSVM::init(){
  _alpha.clear();
}
void sparseKernelSVM::setParameter(matd &par){
  for(size_t i=0;i<_k;++i){
    for(size_t j=0; j< par[i].size(); ++j){
      if(par[i][j] != 0)
        _alpha.insert(i,j,par[i][j]);
    }
  }
}
