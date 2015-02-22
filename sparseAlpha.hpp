#ifndef _sparseAlpha__
#define _sparseAlpha__
#include "def.hpp"
#include "Kernel.hpp"
#include <set>
#include <map>

typedef map<size_t,double> myMapD;
typedef map<size_t,double>::iterator myMapD_iter;
typedef map<size_t,unsigned int> myMapUI;
typedef map<size_t,unsigned int>::iterator myMapUI_iter;
typedef set<size_t> mySet;
typedef set<size_t>::iterator mySet_iter;

class sparseAlpha{
public:
  sparseAlpha(size_t k, size_t p);
  sparseAlpha(matd m);
  void insert(size_t row,size_t col, double val);
  void insert(size_t col, const vec &v); // if col(i) == 0 remove this cell.

  void remove(size_t row,size_t col);
  // preform res(k) = scalar * alpha(k,p) * vec(p)
  size_t vecMul(vec & res, double scalar,Kernel * ker, size_t col,
                           vector<map<size_t,double>::iterator> & indx);
  size_t vecMul(vec & res, double scalar,Kernel * ker, vec & v);
  void setK(size_t k);
  void setP(size_t p);

  void plus(sparseAlpha alpha); //add two objects;
  double norm(Kernel* ker);//calc the norm with respect to kernel: |alpha^t K alpha |_1
  void write(FILE* pFile);
protected:
  size_t _k;
  size_t _p;
  vector<myMapD> _alpha;
  
  

};
#endif
