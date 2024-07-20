#include <iostream>
#include "../include/dotlis.h"
#include "../include/mlp.h"

using namespace dotlis;

using namespace std;

int main(){
    Dataframe df("Flood_dataset.txt", 3);
    cout << df.get(0, 4);
    return 0;
}
