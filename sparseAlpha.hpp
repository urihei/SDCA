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
  void remove(size_t row,map<size_t,double>::iterator col);
  void set(sparseAlpha & other);
  // preform res(k) = scalar * alpha(k,p) * vec(p)
  size_t vecMul(vec & res, double scalar,Kernel * ker, size_t col,
                vector<map<size_t,double>::iterator> & indx);
  size_t vecMul(vec & res, double scalar,Kernel * ker, size_t col,
                vector<map<size_t,double>::iterator> & indx,bool includeSelf);
  size_t vecMul(vec & res, double scalar,Kernel * ker, vec & v);
  size_t vecMul(vec & res, double scalar,Kernel * ker, double* v);
  void updateAcc(double beta,sparseAlpha &pr,sparseAlpha & nA);
  void setK(size_t k);
  void setN(size_t n);
  bool isIn(size_t row, size_t col);
  myMapD_iter getEnd(size_t k);
 
  void add(sparseAlpha alpha); //add two objects;
  double norm(Kernel* ker,vec & res);
  double norm(Kernel* ker);//calc the norm with respect to kernel: |alpha^t K alpha |_1
  string toString();
  void write(FILE* pFile);
  void clear();
protected:
  size_t _k;
  size_t _n;
  vector<myMapD> _alpha;
  
  

};
#endif
