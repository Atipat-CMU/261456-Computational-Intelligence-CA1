#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>

using namespace std;

#include "Layer.h"
#include "Dataframe.h"
#include "LayerInfo.h"
#include "Parameter.h"

using namespace dotlis;

namespace mlp {
    class Network
    {
        private:
            Layer *input_ly, *output_ly;
            vector<Layer*> hidden_lys;
            void forward(vector<double>& inputs);
            void backward(vector<double>& outputs, double lr);
            Parameter parameter;

        public:
            Network();
            Network(vector<layer_info> layers);
            ~Network();

            void info();
            void fit(Dataframe X, Dataframe y, int epoch, double lr);
    };

    Network::Network()
    {
    }

    Network::Network(vector<layer_info> layers){
        this->parameter = Parameter(layers);
        for(layer_info l_info : layers){
            Layer* layer = new Layer(l_info.type == HIDDEN, l_info.N_node, l_info.activation);
            if(l_info.type == INPUT){
                input_ly = layer;
            }else if(l_info.type == OUTPUT){
                output_ly = layer;
            }else{
                hidden_lys.push_back(layer);
            }
        }

        if(!hidden_lys.empty()){
            hidden_lys[0]->connect(input_ly, parameter, 1);
            for(int i = 1; i < hidden_lys.size(); i++){
                hidden_lys[i]->connect(hidden_lys[i-1], parameter, i+1);
            }
            output_ly->connect(hidden_lys[hidden_lys.size()-1], parameter, hidden_lys.size()+1);
        }else{
            output_ly->connect(input_ly, parameter, 1);
        }
    }

    Network::~Network(){
        delete input_ly;
        delete output_ly;
        for (Layer* layer : hidden_lys) {
            delete layer;
        }
    }

    void Network::info(){
        for(int i = 1; i <= hidden_lys.size(); i++){
            cout << printf("layer %d : ", i);
            cout << &hidden_lys[i-1];
        }
    }

    void Network::forward(vector<double>& inputs){
        input_ly->set_input(inputs);
        for(Layer* ly : hidden_lys){
            ly->forward();
        }
        output_ly->forward();
    }

    void Network::backward(vector<double>& outputs, double lr){
        for(Layer* ly : hidden_lys){
            ly->updateGrad(outputs);
        }
        output_ly->updateGrad(outputs);
        output_ly->backprop(lr);
        for(auto it = hidden_lys.rbegin(); it != hidden_lys.rend(); ++it){
            (*it)->backprop(lr);
        }
    }

    void Network::fit(Dataframe X, Dataframe y, int epoch, double lr){
        if(X.get_width() != input_ly->size()){
            throw runtime_error("Input size not match");
        }
        if(y.get_width() != output_ly->size()){
            throw runtime_error("Output size not match");
        }
        while(epoch--){
            vector<double> inputs = X.getRow(0);
            vector<double> outputs = y.getRow(0);

            this->forward(inputs);
            cout << output_ly->get_output()[0] << endl;
            this->backward(outputs, lr);
        }
    }
}

#endif
