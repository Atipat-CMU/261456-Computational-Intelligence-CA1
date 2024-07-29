#ifndef NEURAL_H
#define NEURAL_H

#include <vector>

namespace mlp {
    class Neural
    {
        private:
            double (*activation)(double, bool);
            double v, y = 0;

        public:
            Neural();
            Neural(double y);
            Neural(double (*fun)(double, bool));
            ~Neural();

            void update(double v);
            void setY(double y);
            double getY();
    };

    Neural::Neural()
    {
    }

    Neural::Neural(double y){
        this->y = y;
    }

    Neural::Neural(double (*fun)(double, bool)){
        this->activation = fun;
        this->y = 0;
        this->v = 0;
    }

    Neural::~Neural()
    {
    }

    void Neural::update(double v){
        this->v = v;
        this->y = activation(v, true);
    }

    void Neural::setY(double y){
        this->y = y;
    }

    double Neural::getY(){
        return y;
    }
}

#endif
