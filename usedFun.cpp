#include "usedFun.hpp"

boost::mt19937 gen(999);
void initRand(){
  gen.seed(time(NULL));//999);//
}
//numbers in [0,d] includ both numbers
int roll(unsigned int d){
  boost::random::uniform_int_distribution<> dist(0, d);
  return  dist(gen);
}

void randperm(unsigned int n,ivec &arr, ivec &prePrm){
    arr[0] = prePrm[0];
    for(size_t i =1; i<n;i++){
        unsigned int d = roll(i);
        arr[i] = arr[d];
        arr[d] = prePrm[i];
    }
}
void cumsum(double* a, size_t len,double* b){
    if(len == 0)
        return;
    b[0] = a[0];
    for(size_t i=1;i<len;++i){
        b[i]=b[i-1]+a[i];
    }
}
/**
 * Compare the last element between com[len-1]<= a[len-1] <= 1.
 **/
size_t findFirstBetween(double* a,double* com, size_t len){
    if(len == 0)
        return -1;
    
    size_t i = 0;
    while((i<len-1)&&((a[i] < com[i]) || (a[i] > com[i+1])))
        i++;
    if(i==len-1){
        if((a[i]>=com[i])&& (a[i]<= 1)){
            return i;
        }
        return len;
    }
    return i;
}
size_t findFirst(double* a,size_t len){
    if(len == 0)
        return -1;
    size_t i=0;
    while((i<len) && a[i] != 1) 
        i++;
    return i;
}

void fillMatrix(matd &data1, mat &data2){
    size_t n = data1.size();
    if(n == 0)
        return;
    size_t p =  data1[0].size();
    
    data2.resize(n,p);
    for(size_t i=0;i<n;++i){
      /* memory inefficient
        for(size_t j=0;j<p;++j){
            data2(i,j) = data1[i][j];
        }
      */
      data2.row(i) = VectorXd::Map(&data1[i][0],p);
    }
}
unsigned long int prod(unsigned int s, unsigned int e){
    unsigned int ret = 1;
    while(s<=e){
        ret *= s;
        s++;
    }
    return ret;
}
/**
 *
 **/
 double sumLog(unsigned int s, unsigned int e){
    double ret = 0;
    while(s<=e){
        ret += log(s);
        s++;
    }
    return ret;
}
/**
 * Calc multinomial coefficient /2^N
 **/
unsigned long long int multinomial(vector<unsigned int> &v){
    return round(exp(logMultinomial(v)));
}
double logMultinomial(vector<unsigned int> &v){
    size_t l = v.size();
    std::sort(v.begin(),v.end(),std::greater<unsigned int>());
    double ret = 0; 
    unsigned int k = v[0];
    for(size_t d=1; d<l; ++d){
        ret  += sumLog(k+1,k+v[d])-sumLog(2,v[d]);
        //        cerr<<k<< " "<<v[d]<< " "<<ret<< " ";
        k += v[d];
    }
    //    cerr<<endl;
    return ret;
}
void ReadTrainData(string fileName,matd& data,ivec & label,vector<int> & label_map){
    string line;
    ifstream myfile;
    myfile.open(fileName.c_str(),ifstream::in);
    double tmp;
    map<int,size_t> lMap;
    unsigned int countLabel = 0;
    if (!myfile.is_open()){
        cerr << "Unable to open file" << fileName<<endl; 
        assert(false);
    }
    while ( getline (myfile,line) ){
        vec v;
        istringstream iss(line);
        while(iss){
            iss>>tmp;
            if (iss)
                v.push_back(tmp);
        }
        int cl = (int)(v.back());

        map<int,size_t>::iterator lb = lMap.lower_bound(cl);
        if(lb != lMap.end() && !(lMap.key_comp()(cl, lb->first))){
            label.push_back(lb->second);
        }else{
            label.push_back(countLabel);
            lMap.insert(lb, map<int,size_t >::value_type(cl, countLabel));
            countLabel++;
        }

        v.pop_back();

        data.push_back(v);
    }
    label_map.resize(countLabel);
    for(map<int,size_t>::iterator it = lMap.begin(); it != lMap.end();++it){
        label_map[it->second] = it->first;
    }
    myfile.close();
}
size_t ReadTrainData(string fileName,double* &data,size_t* & label_arr,int* & label_map,size_t &n,size_t &p){
    string line;
    ifstream myfile;
    myfile.open(fileName.c_str(),ifstream::in);
    double tmp;
    map<int,size_t> lMap;
    unsigned int countLabel = 0;
    if (!myfile.is_open()){
        cerr << "Unable to open file" << fileName<<endl; 
        assert(false);
    }
    getline (myfile,line);
    istringstream iss(line);
    p = 0;
    //counting columns
    while(iss){
      iss>>tmp;
      if (iss)
        p++;
    }
    //countnig rows
    p--; // the last coulumn is a label.
    n=1;
    while ( getline (myfile,line) ){
      n++;
    }
    //return the stream to the begining of the file.
    myfile.clear();
    myfile.seekg(0,myfile.beg);
    data = new double[n*p];
    vector<size_t> label;
    size_t ind_n = 0;
    
    while ( getline (myfile,line) ){
        istringstream iss(line);
        for(size_t ind_p =0; ind_p < p; ++ind_p){
            iss>>tmp;
            if (iss){
              data[ind_p+ind_n*p] = tmp;
            }
        }
        int cl;
        iss >> cl;

        map<int,size_t>::iterator lb = lMap.lower_bound(cl);
        if(lb != lMap.end() && !(lMap.key_comp()(cl, lb->first))){
            label.push_back(lb->second);
        }else{
            label.push_back(countLabel);
            lMap.insert(lb, map<int,size_t >::value_type(cl, countLabel));
            countLabel++;
        }
        ind_n++;
    }
    label_arr = new size_t[n];

    ind_n =0;
    for(vector<size_t>::iterator it = label.begin(); it != label.end();++it)
      label_arr[ind_n++] = *it;
    
    label_map = new int[countLabel];
    for(map<int,size_t>::iterator it = lMap.begin(); it != lMap.end();++it){
        label_map[it->second] = it->first;
    }
    myfile.close();
    return countLabel;
}
double calcNormFeature(double squeredNorm,double max_norm){
  return 1 - squeredNorm/max_norm;
}
double AddNormAsFeature(matd &data){

  double max_norm = -1;
  size_t N =  data.size();
  vec norm(N);
  
  for(size_t i=0; i<N; ++i){
    norm[i] = 0;
    for(size_t j=0; j< data[i].size(); ++j)
      norm[i] += data[i][j] * data[i][j];

    //norm[i] = sqrt(norm[i]);
    max_norm = (norm[i] > max_norm)? norm[i]:max_norm;
  }
  for(size_t i=0; i<N;++i){
    data[i].push_back(calcNormFeature(norm[i],max_norm));
  }
  return max_norm;
}

double normalizeData(matd &data, vec &meanVec){
  size_t n = data.size();
  size_t p = data[0].size();
  meanVec.resize(p);
  std::fill(meanVec.begin(),meanVec.end(),0);
  for(vector<vector<double>>::iterator it = data.begin(); it != data.end(); ++it){
    for(size_t i=0; i<p; ++i){
      meanVec[i] += (*it)[i];
    }
  }
  for(size_t i = 0; i<p; ++i){
    meanVec[i] /= (n+0.0);
  }
  for(vector<vector<double>>::iterator it = data.begin(); it != data.end(); ++it){
    for(size_t i=0; i<p; ++i){
      (*it)[i] -= meanVec[i];
    }
  }
  double maxNorm  = 0;
  for(vector<vector<double>>::iterator it = data.begin(); it != data.end(); ++it){
    double norm = 0;
    for(size_t i=0; i<p; ++i){
      norm += (*it)[i]*(*it)[i];
    }
   if (maxNorm<norm){
     maxNorm = norm;
   }
  }
  maxNorm = sqrt(maxNorm);
  for(vector<vector<double>>::iterator it = data.begin(); it != data.end(); ++it){
    for(size_t i=0; i<p; ++i){
      (*it)[i] /= maxNorm;
    }
  }
  return maxNorm;
}
void normalizeData(matd &data, vec &meanVec, double maxNorm){
  size_t p = data[0].size();
  for(vector<vector<double>>::iterator it = data.begin(); it != data.end(); ++it){
    for(size_t i=0; i<p; ++i){
      (*it)[i] -= meanVec[i];
    }
  }
  for(vector<vector<double>>::iterator it = data.begin(); it != data.end(); ++it){
    for(size_t i=0; i<p; ++i){
      (*it)[i] /= maxNorm;
    }
  }
}
double normalizeData(double* data, double* meanVec, size_t n,size_t p){
  std::fill(meanVec,meanVec+p,0);
  
  for(size_t i=0; i<n;++i){
    for(size_t j=0; j<p; ++j){
      meanVec[j] += data[i*p+j];
    }
  }
  
  for(size_t i = 0; i<p; ++i){
    meanVec[i] /= (n+0.0);
  }
  
  for(size_t i=0; i<n;++i){
    for(size_t j=0; j<p; ++j){
      data[i*p+j] -= meanVec[j];
    }
  }
  
  double maxNorm  = 0;
  for(size_t i=0; i<n;++i){
    double norm = 0;
    for(size_t j=0; j<p; ++j){
      norm += data[i*p+j]*data[i*p+j];
    }
    if (maxNorm<norm){
      maxNorm = norm;
    }
  }
  maxNorm = sqrt(maxNorm);
  for(size_t i=0; i<n*p;++i){
      data[i] /= maxNorm;
  }
  return maxNorm;
}

void normalizeData(double* data, double* meanVec, double maxNorm, size_t n,size_t p){
  for(size_t i=0; i<n;++i){
    for(size_t j=0; j<p; ++j){
      data[i*p+j] -= meanVec[j];
    }
  }
  for(size_t i=0; i<n*p;++i){
      data[i] /= maxNorm;
  }
}
