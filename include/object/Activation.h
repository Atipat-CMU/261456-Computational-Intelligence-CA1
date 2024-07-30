#include <math.h>

namespace mlp {
    double sigmoid(double x, bool isForward){
        if(isForward){
            return 1/(1+exp(-x));
        }else{
            double f = sigmoid(x, true);
            return f*(1-f);
        }
    }

    double linear(double x, bool isForward){
        if(isForward){
            return x;
        }else{
            return 1;
        }
    }
}
