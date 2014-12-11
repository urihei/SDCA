#include "learnSVM.hpp"

learnSVM::learnSVM(ivec &y,matd &data, unsigned int iter,
                   unsigned int accIter,double lambda)
    :_iter(iter),_accIter(accIter),_lambda(lambda),_y(y){
    _n = data.size();
    size_t p =  data[0].size();
    _data.resize(_n,p);
    
    for(size_t i=0;i<_n;++i){
        for(size_t j=0;j<p;++j){
            _data(i,j) = data[i][j];
        }
    }

}
learnSVM::~learnSVM(){
}
void learnSVM::setIter(unsigned int iter){
    _iter = iter;
}
void learnSVM::setAccIter(unsigned int iter){
    _accIter = iter;
}
void learnSVM::setLambda(double lambda){
    _lambda = lambda;
}
unsigned int learnSVM::getIter(){
    return _iter;
}
unsigned int learnSVM::getAccIter(){
    return _accIter;
}
double learnSVM::getLambda(){
    return _lambda;
}
mat* learnSVM::getData(){
    return &_data;

}
