#include <iostream>
#include "../../../../include/dotlis.h"
#include "../../../../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df("../../Flood_dataset.txt", 3);
    df.to_csv("dataset.csv");
    Dataframe df_random = df.random();

    int k = 10;
    vector<Dataframe> fold_ls = df_random.split_n(k);

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, sigmoid, 8},
        {OUTPUT, sigmoid, 1},
    };

    vector<History> history_ls;
    Parameter init_parameter = param_read("../default.param");

    double sum_rmse_t = 0, sum_rmse_v = 0;
    int N_t = 0, N_v = 0;

    Normalizer normalizer(0, 700, 0, 1);

    for(int i = 0; i < k; i++){
        Dataframe df_train;
        Dataframe df_validate;

        for(int j = 0; j < fold_ls.size(); j++){
            if(i != j){
                df_train.extend(fold_ls[j]);
            }else{
                df_validate.extend(fold_ls[j]);
            }
        }

        Dataframe X_train = df_train.get_column_without({8});
        Dataframe y_train = df_train.get_column({8});
        Dataframe X_train_norm = normalizer.forward(X_train);
        Dataframe y_train_norm = normalizer.forward(y_train);

        Dataframe X_val = df_validate.get_column_without({8});
        Dataframe y_val = df_validate.get_column({8});
        Dataframe X_val_norm = normalizer.forward(X_val);

        Network network(layers);
        network.setParam(init_parameter);

        History history = network.fit(X_train_norm, y_train_norm, 2000, 10, 0.2);
        history_ls.push_back(history);

        Dataframe y_pred_train_norm = network.predict(X_train_norm);
        Dataframe y_pred_train = normalizer.backward(y_pred_train_norm);
        Dataframe y_pred_val_norm = network.predict(X_val_norm);
        Dataframe y_pred_val = normalizer.backward(y_pred_val_norm);

        cout << "fold-" << i+1 << ":  ";
        double rmse_t = calRMSE(y_train, y_pred_train);
        double rmse_v = calRMSE(y_val, y_pred_val);
        cout << "RMSE(train): " << rmse_t << "\t";
        cout << "RMSE(validate): " << rmse_v << "\t";
        cout << "(" << rmse_v - rmse_t << ")\n";
        sum_rmse_t += rmse_t; N_t++;
        sum_rmse_v += rmse_v; N_v++;

        string exp_name = "2000_mlp_8_lr10_m0.2";

        history.exportError("error/error" + exp_name + "_f" + to_string(i+1) + ".csv");
        network.getParam().to_file("parameter/" + exp_name + "_f" + to_string(i+1) + ".param");
    }
    cout << "---------------------------------------------------------" << endl;
    cout << "Average RMSE(train): " << sum_rmse_t/N_t << endl;
    cout << "Average RMSE(validate): " << sum_rmse_v/N_v << endl;

    return 0;
}
