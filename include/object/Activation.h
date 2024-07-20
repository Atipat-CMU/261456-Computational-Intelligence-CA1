#include <math.h>

typedef long long ll;

namespace mlp {
    int sigmoid(ll x, bool isForward){
        if(isForward){
            return 1/(1+exp(-x));
        }else{
            ll f = sigmoid(x, true);
            return f*(1-f);
        }
    }
}
