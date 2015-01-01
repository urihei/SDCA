#include "baseKernelSVM.hpp"
baseKernelSVM::baseKernelSVM(size_t k, double lambda, double gamma , unsigned int iter, unsigned int accIter):svm(k,lambda,gamma,iter,accIter){}

double baseKernelSVM::getGap(mat &alpha,mat &zALPHA){
    double pr = 0.0;
    double du = 0.0;

    double lambdaN  = 1/(_lambda * _n);

    mat az = zALPHA + alpha;
    ArrayXd a(_k);

    double normPart = 0.0;
    
    VectorXd kerCol(_n);    
    ArrayXd b(_k);
    
    for(size_t i = 0; i<_n;++i){
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
    pr = pr/_n+_lambda*normPart;
    du /= _n;
    double gap = pr - du;
    fprintf(stderr,"primal %g\t dual %g\t Gap %g \n",pr,du,gap);
    //    cerr<< "primal:  "<<pr<<"\t dual"<<du<<"\t Gap: "<<gap<<endl;
    return gap;
}
double baseKernelSVM::getGap(){
    mat zAlpha(_k,_n);
    zAlpha.setZero();
    return getGap(_alpha,zAlpha);
}


void baseKernelSVM::learn_acc_SDCA(){
    double kappa = 100*_lambda;
    double mu = _lambda/2;
    double rho = mu+kappa;
    double eta = sqrt(mu/rho);
    double beta = (1-eta)/(1+eta);

    mat alpha(_k,_n);
    alpha.setZero();
    mat zALPHA(_k,_n);
    zALPHA.setZero();
    mat ALPHA_t(_k,_n);
    ALPHA_t.setZero();
    MatrixXd zALPHA_t(_k,_n)
        ;
    double gap = _eps + 1.0;

    double OnePetaSquare = 1+1/eta*1/eta;
    double xi = OnePetaSquare * (1-_gamma/(2*(_k-1)));
    eta = eta /2;
    
    for(unsigned int t =1;t<=_accIter;++t){
        learn_SDCA(alpha,zALPHA,eta/OnePetaSquare * xi);
        zALPHA_t = zALPHA;
        zALPHA = (1+beta)*(zALPHA + alpha) - beta * ALPHA_t;
        ALPHA_t = zALPHA_t+alpha;
        if(t%_chackGap ==0){
            cerr<<"ACC iter: "<<t<<"\t";
            _alpha = ALPHA_t;
            gap = getGap();
        }
        if(gap < _eps)
            break;
        xi = xi * (1-eta);
    }
    _alpha = ALPHA_t;
}

void baseKernelSVM::saveModel(string fileName){}
