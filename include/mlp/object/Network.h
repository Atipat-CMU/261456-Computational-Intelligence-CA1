#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <ctime>

using namespace std;

#include "Layer.h"
#include "../../dotlis/object/Dataframe.h"
#include "LayerInfo.h"
#include "Parameter.h"
#include "History.h"

using namespace dotlis;

namespace mlp {
    class Network
    {
        private:
            Layer *input_ly, *output_ly;
            vector<Layer*> hidden_lys;
            Parameter parameter;
            void forward(vector<double>& inputs);
            void backward(vector<double>& outputs, double lr);
            void updateParam();

        public:
            Network();
            Network(vector<layer_info> layers);
            ~Network();

            void info();
            History fit(Dataframe X, Dataframe y, int epoch, double lr);
    };

    Network::Network()
    {
    }

    Network::Network(vector<layer_info> layers){
        this->parameter = Parameter(layers);
        int ly_count = 0;
        for(layer_info l_info : layers){
            Layer* layer = new Layer(ly_count, l_info.type == HIDDEN, l_info.N_node, l_info.activation);
            if(l_info.type == INPUT){
                input_ly = layer;
            }else if(l_info.type == OUTPUT){
                output_ly = layer;
            }else{
                hidden_lys.push_back(layer);
            }
            ly_count++;
        }

        if(!hidden_lys.empty()){
            hidden_lys[0]->connect(input_ly, parameter);
            for(int i = 1; i < hidden_lys.size(); i++){
                hidden_lys[i]->connect(hidden_lys[i-1], parameter);
            }
            output_ly->connect(hidden_lys[hidden_lys.size()-1], parameter);
        }else{
            output_ly->connect(input_ly, parameter);
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

    // void Network::updateParam(){
    //     if(!hidden_lys.empty()){
    //         hidden_lys[0]->connect(input_ly, parameter);
    //         for(int i = 1; i < hidden_lys.size(); i++){
    //             hidden_lys[i]->connect(hidden_lys[i-1], parameter);
    //         }
    //         output_ly->connect(hidden_lys[hidden_lys.size()-1], parameter);
    //     }else{
    //         output_ly->connect(input_ly, parameter);
    //     }
    // }

    History Network::fit(Dataframe X, Dataframe y, int epoch, double lr){
        if(X.get_width() != input_ly->size()){
            throw runtime_error("Input size not match");
        }
        if(y.get_width() != output_ly->size()){
            throw runtime_error("Output size not match");
        }

        vector<double> error_ls;

        while(epoch--){
            srand(time(0));

            vector<int> index_ls;
            for(int i = 0; i < X.get_depth(); i++){
                index_ls.push_back(i);
            }

            double error = 0;
            while(!index_ls.empty()){
                int range = (index_ls.size() - 1) + 1;
                int num = rand() % range;

                int index = index_ls[num];
                vector<double> inputs = X.getRow(index);
                vector<double> outputs = y.getRow(index);

                this->forward(inputs);

                vector<double> y_ls = output_ly->get_output();
                double sse = 0;
                for(int i = 0; i < y_ls.size(); i++){
                    sse += pow(outputs[i] - y_ls[i], 2);
                }

                error += sse/2.0;

                this->backward(outputs, lr);
                index_ls.erase(index_ls.begin() + index);
            }

            error_ls.push_back(error/X.get_depth());
        }

        return History(error_ls);
    }
}

#endif
