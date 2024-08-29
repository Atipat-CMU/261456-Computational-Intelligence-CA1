#include <iostream>
#include "../../../include/dotlis.h"
#include "../../../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df = read_pat("cross.pat");
    df.to_csv("dataset2.csv");
    Dataframe df_random = df.random();

    int k = 10;
    vector<Dataframe> fold_ls = df_random.split_n(k);

    vector<layer_info> layers = {
        {INPUT, nullptr, 2},
        {HIDDEN, sigmoid, 32},
        {OUTPUT, sigmoid, 2},
    };

    vector<History> history_ls;
    Parameter init_parameter(layers, -1, 1);

    double sum_acc_t = 0, sum_acc_v = 0;
    int N_t = k, N_v = k;

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

        Dataframe X_train = df_train.get_column_without({2,3});
        Dataframe y_train = df_train.get_column({2,3});

        Dataframe X_val = df_validate.get_column_without({2,3});
        Dataframe y_val = df_validate.get_column({2,3});

        Network network(layers);
        network.setParam(init_parameter);

        History history = network.fit(X_train, y_train, 2000, 0.5, 0.9);
        history_ls.push_back(history);

        Dataframe y_pred_train = network.predict(X_train);
        Dataframe y_pred_val = network.predict(X_val);

        y_pred_train = markMax(y_pred_train);
        y_pred_val = markMax(y_pred_val);
        cout << "---------------------------------------------------------" << endl;
        cout << "fold-" << i+1 << ":  " << endl;

        sum_acc_t += calConfusionM(y_pred_train.get_column({0}), y_train.get_column({0}));
        sum_acc_v += calConfusionM(y_pred_val.get_column({0}), y_val.get_column({0}));

        string exp_name = "2000_mlp_32_lr0.1_m0.9";

        history.exportError("error/error" + exp_name + "_f" + to_string(i+1) + ".csv");
        network.getParam().to_file("parameter/" + exp_name + "_f" + to_string(i+1) + ".param");
    }
    cout << "---------------------------------------------------------" << endl;
    cout << "Average Accuracy(train): " << sum_acc_t/N_t << endl;
    cout << "Average Accuracy(validate): " << sum_acc_v/N_v << endl;

    return 0;
}
