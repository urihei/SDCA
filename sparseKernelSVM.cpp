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
  matd pOld(_k);
  for(size_t k=0;k<_k;++k){
    pOld[k] = vec(_n,0);
  }
  return learn_SDCA(_alpha,pOld,_eps);
}
double sparseKernelSVM::learn_SDCA(sparseAlpha &alpha,matd &pOld,double eps){
    double lambdaN = 1/(_n*_lambda);
    
    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;

    ivec prm(_usedN);
    vec kerCol(_n);
    
    
    vec p(_k);
    vec mu(_k);
    vec a(_k);

    vector<map<size_t,double>::iterator> indx(_k);
    double gap = eps + 1;
    for(unsigned int t=1;t <= _iter;++t){
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
          indx[k]->second = -a[k];
        }
        indx[curLabel]->second = norm1A;
        
        if( t % (_usedN* _checkGap) == 0){
            gap = getGap(alpha,pOld);
            
        }
        
        if(gap < eps)
            break;
        
        ind++;
    }

    if(gap >eps){
        gap = getGap(alpha,pOld);
    }
    return gap;
}

double sparseKernelSVM::getGap(sparseAlpha &alpha,matd &pOld){
  double pr = 0.0;
  double du = 0.0;

  double lambdaN  = 1/(_lambda * _n);

  vec a(_k);


  double normPart = 0.0;

  vec kerCol(_n);    
  vec b(_k);
  vector<map<size_t,double>::iterator> indx(_k);

  for(size_t ii = 0; ii<_usedN;++ii){
    size_t i = _prmArray[ii];
    size_t currentLabel = _y[i];
      
    // getCol(i,kerCol);//_ker->dot(i,kerCol);
    // a = lambdaN * az * kerCol;
    alpha.vecMul(a,lambdaN,_ker,i,indx);
    double sumTmp = 0; //(a * alpha.col(i).array()).sum()
    double normSa = 0; //|a|^2 
    double norm1alpha = 0; //|alpha|_1 - c'*alpha
    double normSalpha = 0; //|alpha|^2
    for(size_t k=0;k<_k;++k){
      a[k] += pOld[k][i];
      sumTmp += a[k] * indx[k]->second;
      //    a = (a - a(currentLabel)  + 1)/_gamma;
      a[k] = (a[k]-a[currentLabel]+1)/_gamma;
      normSa += a[k]*a[k];
      norm1alpha += indx[k]->second;
      normSalpha += indx[k]->second*indx[k]->second;
    }
    normSa -= (1.0/_gamma) *(1.0/_gamma); // the value of a[currentLabel] needed to be removed.
    
    norm1alpha -= indx[currentLabel]->second;
    a[currentLabel] = 0;
     
    
    normPart += lambdaN*sumTmp;
        
    //b = b - a;
    double normBs = project_SDCA(a,b);
    
    pr += _gamma/2 * (normSa - normBs);
    du -= norm1alpha + _gamma/2 *
      (normSalpha - indx[currentLabel]->second * indx[currentLabel]->second );
  }
  pr = pr/_n+_lambda*normPart;
  du /= _n;
  double gap = pr - du;
  if(_verbose)
    fprintf(stderr,"primal %g\t dual %g\t Gap %g \n",pr,du,gap);
  return gap;
}


void sparseKernelSVM::classify(matd &data, ivec &res){
    size_t n = data.size();
    vec ya(_k);
    vec kerCol(_n);
    vec empty;
    cerr<<"Start kernel classify"<<endl;
    for(size_t i=0;i<n;++i){
      //_ker->dot(data[i],kerCol);
      // ya = _alpha * kerCol;
      // ya.maxCoeff(&index);
      //  res[i] = (size_t) index;
      res[i] = _alpha.vecMul(ya,1/_lambda,data[i],_ker,empty); //build a new function that recive vector and not column.
    }
    
}
void sparseKernelSVM::classify(ivec_iter &itb,ivec_iter &ite,ivec &res){
  size_t n = std::distance(itb,ite);
  if(res.size() != n){
    res.resize(n);
  }
  vec kerCol(_n);
  size_t i =0;
  for(ivec_iter it =itb; it<ite;++it){
    _ker->dot(*it,kerCol);
    (_alpha * kerCol).array().maxCoeff(&index);
    res[i++] = (size_t) index;
  }
}
void sparseKernelSVM::classify(const Ref <const MatrixXd> &data,ivec &res){
    size_t n = data.cols();
    VectorXd ya(_k);
    VectorXd kerCol(_n);
    MatrixXf::Index index;
    
    for(size_t i=0;i<n;++i){
        _ker->dot(data.col(i),kerCol);
        ya = _alpha * kerCol;
        ya.maxCoeff(&index);
        res[i] = (size_t) index;
    }
}
void sparseKernelSVM::saveModel(FILE* pFile){
    saveModel(pFile,_ker->getName(),_alpha);
}
void sparseKernelSVM::getCol(size_t i,Ref <VectorXd>  kerCol){
    _ker->dot(i,kerCol);
}
