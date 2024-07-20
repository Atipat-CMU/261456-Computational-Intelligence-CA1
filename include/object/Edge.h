#ifndef EDGE_H
#define EDGE_H

#include "Neural.h"

typedef long long ll;

namespace mlp {
    class Edge
    {
        private:
            Neural *head, *tail;
            ll weight;

        public:
            Edge();
            ~Edge();

            Neural* getHead();
            Neural* getTail();
            ll getW();
    };

    Edge::Edge()
    {
    }

    Edge::~Edge()
    {
    }

    Neural* Edge::getHead(){
        return head;
    }

    Neural* Edge::getTail(){
        return tail;
    }

    ll Edge::getW(){
        return weight;
    }
}

#endif
