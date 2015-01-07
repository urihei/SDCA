#include "linearSVM.hpp"
#include "usedFun.hpp"

linearSVM::linearSVM(ivec &y,matd &data,size_t k,
                     double lambda, double gamma,
                     unsigned int iter,unsigned int accIter):svm(k,lambda,gamma,iter,accIter){
    fillMatrix(data,_data);
    _data.transposeInPlace();
    _n = _data.cols();
    _p = _data.rows();
    _y = y;
    _W.resize(_p,_k);
    _squaredNormData.resize(_n);
    for(size_t i = 0;i<_n;++i){
        _squaredNormData(i) = _data.col(i).squaredNorm();
    }
}
double linearSVM::learn_SDCA(){
    mat alpha(_k,_n);
    alpha.setZero();
    mat zW(_p,_k);
    zW.setZero();
    return learn_SDCA(alpha, zW,_eps);
}

double linearSVM::learn_SDCA(mat &alpha, mat &zW){
    return learn_SDCA(alpha, zW,_eps);
}
double linearSVM::learn_SDCA(mat &alpha, mat &zW,double eps){


    double lambdaN = 1/(_n*_lambda);
    double gammaLambdan = _gamma*_n*_lambda;


    unsigned int* prm = new unsigned int[_n];
  

    unsigned int ind = 0;

    ArrayXd p(_k);
    ArrayXd mu(_k);
    ArrayXd a(_k);
    double C;
  
  
    _W = zW + lambdaN * _data *alpha.transpose(); 

    double gap = eps + 1;
    
    for(unsigned int t=1;t<=_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
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

        if( t % (_n* _checkGap) == 0){
            gap = getGap(alpha,zW);
            
        }
        
        if(gap < eps)
            break;

        ind++;
    }
    delete prm;
    return gap;
}

void linearSVM::learn_acc_SDCA(){
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
	  cerr<<"ACC iter: "<<t<<" gap: ";
	  gap = (1+rho/mu)*epsilon_t + 
	    (rho*kappa)/(2*mu)*(_W-zW).squaredNorm();
	  cerr<<gap<<"\t";
	}
        if(gap < _eps)
	  break;
        zW = (1+beta)*_W - beta * W_t;
        W_t = _W*1;
        xi = xi * (1-eta);
    }
}

double linearSVM::getGap(mat &alpha, mat &zW){
    double pr = 0.0;
    double du = 0.0;

    ArrayXd a(_k);
    ArrayXd b(_k);
    
    for(size_t i = 0; i<_n;++i){
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
    pr = pr/_n + _lambda * (_W.array() * (_W.array() - zW.array())).sum();
    du /= _n;
    double gap = pr - du;
    if(_verbose)
        cerr<< "primal: "<<pr<<"\t dual: "<<du<<"\t Gap: "<<gap<<endl;
    return gap;
}
double linearSVM::getGap(){
    mat zW(_p,_k);
    zW.setZero();
    mat alpha(_k,_n);
    alpha.setZero();
    return getGap(alpha,zW);
}
void linearSVM::classify(matd &data,ivec &res){
    MatrixXd mData;
    fillMatrix(data,mData);
    cerr<<"Data rows: "<<mData.rows()<<" cols: "<<mData.cols()<<endl;
    mData.transposeInPlace();
    size_t n = mData.cols();
    MatrixXd ya(_k,n);
    cerr<<"W p: "<<_W.rows() <<"W k"<<_k<<endl;
    ya = _W.transpose()* mData;
    MatrixXf::Index index;
    for(size_t i=0;i<n;i++){
        ya.col(i).maxCoeff(&index);
        res[i] = (size_t) index;
    }
}
void linearSVM::saveModel(string fileName){
    saveModel(fileName,"Linear",_W);
}
void linearSVM::setParameter(matd &par){
    fillMatrix(par,_W);
}
