#include <iostream>
#include "../../../include/dotlis.h"
#include "../../../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df("../Flood_dataset.txt", 3);
    Dataframe df_random = df.random();

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, linear, 32},
        {OUTPUT, linear, 1},
    };

    vector<History> history_ls;
    Parameter pretrain_parameter = param_read("2000_mlp_16^4_lr1e-7_m0.2_f10.param");

    Dataframe X_val = df_random.get_column_without({8});
    Dataframe y_val = df_random.get_column({8});

    Network network(layers);
    network.setParam(pretrain_parameter);

    Dataframe y_pred_val = network.predict(X_val);

    double rmse_v = calRMSE(y_val, y_pred_val);
    cout << "RMSE(validate): " << rmse_v << "\t";

    return 0;
}
