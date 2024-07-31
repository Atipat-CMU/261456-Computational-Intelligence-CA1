#include <iostream>
#include "../../../include/dotlis.h"
#include "../../../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df("../Flood_dataset.txt", 3);
    Dataframe df_random = df.random();

    int k = 10;
    vector<Dataframe> fold_ls = df_random.split_n(k);

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, linear, 8},
        {HIDDEN, linear, 8},
        {HIDDEN, linear, 4},
        {HIDDEN, linear, 2},
        {OUTPUT, linear, 1},
    };

    vector<History> history_ls;
    Parameter init_parameter(layers);

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

        Dataframe X_val = df_validate.get_column_without({8});
        Dataframe y_val = df_validate.get_column({8});

        Network network(layers);
        network.setParam(init_parameter);
        History history = network.fit(X_train, y_train, 1000, 1.0E-10, 0.8);
        history_ls.push_back(history);

        cout << history.get_latest_err() << endl;

        // history.exportError("error/error_500_mlp_8_8_4_2_lr1e-10.csv");
        // network.getParam().to_file("parameter/500_mlp_8_8_4_2_lr1e-10.param");
    }

    return 0;
}
