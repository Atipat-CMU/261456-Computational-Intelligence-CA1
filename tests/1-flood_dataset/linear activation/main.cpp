#include <iostream>
#include "../../../include/dotlis.h"
#include "../../../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df("../Flood_dataset.txt", 3);
    df.random();

    Dataframe X_train = df.get_column_without({8});
    Dataframe y_train = df.get_column({8});

    vector<layer_info> layers = {
        {INPUT, nullptr, 8},
        {HIDDEN, linear, 8},
        {HIDDEN, linear, 8},
        {HIDDEN, linear, 4},
        {HIDDEN, linear, 2},
        {OUTPUT, linear, 1},
    };

    Network network(layers);
    History history;
    history = network.fit(X_train, y_train, 500, 1.0E-10, 0.8);
    history.exportError("error/error_500_mlp_8_8_4_2_lr1e-10.csv");
    network.getParam().to_file("parameter/500_mlp_8_8_4_2_lr1e-10.param");

    return 0;
}
