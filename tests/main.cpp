#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df("Flood_dataset.txt", 3);

    Dataframe X_train = df.get_column_without({8});
    Dataframe y_train = df.get_column({8});

    vector<layer_info> layers = {
        {INPUT, linear, 8},
        {OUTPUT, linear, 1},
    };

    Network network(layers);
    network.fit(X_train, y_train, 200, 0.1);

    return 0;
}
