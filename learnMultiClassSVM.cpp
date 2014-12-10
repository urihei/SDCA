#include "learnMultiClassSVM.hpp"
#include "usedFun.hpp"

learnMultiClassSVM::learnMultiClassSVM(ivec &y,matd &data,size_t k,
                                       unsigned int iter,unsigned int accIter,
                                       double lambda)
    :learnSVM(y,data,iter,accIter,lambda),_k(k){
    _gamma = 0.1;
}

void learnMultiClassSVM::learn_SDCA(mat &alpha, mat &zALPHA){
    double lambdaN = 1/(_n*_lambda);

    double gammaLambdan = _gamma*_n*_lambda;


    double C;
    unsigned int ind = 0;
    unsigned int* prm = new unsigned int[_n];
    mat AlTemp(_k,_n);

    ArrayXd p(_k);
    ArrayXd c(_k);
    ArrayXd mu(_k);
    ArrayXd a(_k);

    ArrayXd muh(_k);
    ArrayXd mub(_k);
    ArrayXd z(_k);
    
    ArrayXd OneToK(_k);
    OneToK.setOnes();
    cumsum(OneToK.data(),_k,OneToK.data());
    ArrayXd normOne(_k);
    for(unsigned int t=0;t<_iter;++t){
        if((ind % _n) == 0){
            randperm(_n,prm);
            ind = 0;
        }
        size_t i = prm[ind];
        AlTemp = alpha;
        AlTemp.col(i).setZero();
        AlTemp = AlTemp + zALPHA;

        p = lambdaN * (AlTemp * _data.col(i));
        p = p - p(_y[i]);

        c.setOnes();
        c(_y[i]) = 0;

        
        mu = (c+p)/(_gamma+(_data(i,i)*lambdaN));
        C = 1/(1+(gammaLambdan/_data(i,i)));

        //optimizeDual_SDCA(mu,C,a);

        //creating muh
        muh = mu.max(0);
        sort(muh.data(),muh.data()+_k);
        muh.reverseInPlace();
        //creating mub
        cumsum(muh.data(),_k,mub.data());
        //creating z
        z  = (mub - (OneToK * muh)).min(1);
        //calc normOne (matlab mubDjC)
        normOne = mub/(1+(OneToK*C));
        //find indF
        size_t indF = findFirstBetween(normOne.data(),z.data(),_k);


        if(indF >= _k){
	  size_t indJ = findFirst(z.data(),_k)-1;
           a = (mu+((1-mub(indJ))/(indJ+1))).max(0);
	  if(((a-mu).matrix().squaredNorm()+C) > (mu.matrix().squaredNorm())){
              a.setZero();
	  }
        }else{
            a = (mu+((normOne(indF)-mub(indF))/(indF+1))).max(0);
        }
        // END optimizeDual_SDCA
        alpha.col(i) = -a;
        alpha(_y[i],i) = a.matrix().lpNorm<1>();
                
        ind++;
    }

    delete prm;
    //to remove
    MatrixXd ya(_k,_n);
    ya = lambdaN * (alpha * _data);
    unsigned int  count =0;
    MatrixXf::Index index;
    for(size_t i=0;i<_n;i++){
        ya.col(i).maxCoeff(&index);
        
        if( ((unsigned int)index) != _y[i])
            count++;
    }
    cout<<"The Number of train error "<<count<<endl;

}
void learnMultiClassSVM::acc_learn_SDCA(mat &alpha){

}
void learnMultiClassSVM::returnModel(mat &model){

}
void learnMultiClassSVM::setGamma(double gamma){
    _gamma = gamma;
}
double learnMultiClassSVM::getGamma(){
    return _gamma;
}
