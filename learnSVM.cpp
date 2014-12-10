#include "learnSVM.hpp"

learnSVM::learnSVM(vector<int> &y,vector<vector<double>> &data, unsigned int iter,
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
