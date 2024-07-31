#ifndef WEIGHTLIST_H
#define WEIGHTLIST_H

#include <vector>
#include <random>
#include <ctime>

using namespace std;

#include "LayerInfo.h"

namespace mlp {
    class Parameter
    {
        private:
            vector<vector<double>> weight_lys;
            vector<vector<double>> bias_lys;
            
        public:
            Parameter();
            Parameter(vector<layer_info> layers);
            Parameter(vector<layer_info> layers, double min, double max);
            ~Parameter();

            vector<double> get_weight_ly(int ly);
            vector<double> get_bias_ly(int ly);
    };
    
    Parameter::Parameter()
    {
    }

    Parameter::Parameter(vector<layer_info> layers){
        srand(time(0));

        for(int j = 0; j < layers.size(); j++){
            layer_info curr_ly = layers[j];
            vector<double> weight_ly;
            vector<double> bias_ly;
            if(curr_ly.type != INPUT){
                layer_info prev_ly = layers[j-1];
                for(int i = 0; i < curr_ly.N_node; i++){
                    double max = 1/sqrt(prev_ly.N_node);
                    double min = -1/sqrt(prev_ly.N_node);
                    for(int k = 0; k < prev_ly.N_node; k++){
                        float r1 = (float)rand() / (float)RAND_MAX;
                        weight_ly.push_back(min + r1 * (max - min));
                    }
                    float r2 = (float)rand() / (float)RAND_MAX;
                    bias_ly.push_back(r2 * 1);
                }
            }
            weight_lys.push_back(weight_ly);
            bias_lys.push_back(bias_ly);
        }
    }

    Parameter::Parameter(vector<layer_info> layers, double min, double max){
        srand(time(0));

        for(int j = 0; j < layers.size(); j++){
            layer_info curr_ly = layers[j];
            vector<double> weight_ly;
            vector<double> bias_ly;
            if(curr_ly.type != INPUT){
                layer_info prev_ly = layers[j-1];
                for(int i = 0; i < curr_ly.N_node; i++){
                    for(int k = 0; k < prev_ly.N_node; k++){
                        float r1 = (float)rand() / (float)RAND_MAX;
                        weight_ly.push_back(min + r1 * (max - min));
                    }
                    float r2 = (float)rand() / (float)RAND_MAX;
                    bias_ly.push_back(min + r2 * (max - min));
                }
            }
            weight_lys.push_back(weight_ly);
            bias_lys.push_back(bias_ly);
        }
    }
    
    Parameter::~Parameter()
    {
    }

    vector<double> Parameter::get_weight_ly(int ly){
        return weight_lys[ly];
    }

    vector<double> Parameter::get_bias_ly(int ly){
        return bias_lys[ly];
    }
    
}

#endif
