#ifndef NEURAL_H
#define NEURAL_H

#include <vector>

typedef long long ll;

namespace mlp {
    class Neural
    {
        private:
            int (*activation)(ll, bool);
            ll v, y;

        public:
            Neural();
            Neural(int (*fun)(ll, bool));
            ~Neural();

            void update(int v);
            ll getY();
    };

    Neural::Neural()
    {
    }

    Neural::Neural(int (*fun)(ll, bool)){
        this->activation = fun;
    }

    Neural::~Neural()
    {
    }

    void Neural::update(int v){
        this->v = v;
        this->y = activation(v, true);
    }

    ll Neural::getY(){
        return y;
    }
}

#endif
