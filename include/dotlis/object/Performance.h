#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <cmath>

#include "Dataframe.h"

namespace dotlis{
    double calRMSE(Dataframe y, Dataframe _y){
        if(y.get_depth() != _y.get_depth() 
            || y.get_width() != 1 || y.get_width() != 1){
            runtime_error("input is invalid");
        }
        int N = y.get_depth();
        double sse = 0;
        for(int i = 0; i < N; i++){
            sse += pow(y.get(i,0) - _y.get(i,0), 2);
        }
        return sqrt(sse/N);
    }
}

#endif
