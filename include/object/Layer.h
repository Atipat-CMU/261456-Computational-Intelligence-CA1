#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <unordered_map>

using namespace std;

#include "Neural.h"
#include "Edge.h"

using namespace mlp;

typedef long long ll;

typedef unordered_map<Neural*, vector<Edge*>> edge;

namespace mlp {

    class Layer 
    {
        private:
            vector<Neural*> neurals;
            edge in_edges;

        public:
            Layer();
            ~Layer();

            void forward();
    };

    Layer::Layer()
    {
    }

    Layer::~Layer()
    {
    }

    void Layer::forward(){
        for(Neural *n : neurals){
            vector<Edge*> edges = in_edges[n];
            ll v = 0;
            for(Edge *e : edges){
                v += e->getTail()->getY() * e->getW();
            }
            n->update(v);
        }
    }
}

#endif
