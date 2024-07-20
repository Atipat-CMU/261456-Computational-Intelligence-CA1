#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

namespace dotlis {
    class Dataframe
    {
        private:
            vector<vector<double>> table;

        public:
            Dataframe();
            Dataframe(string file_path, int start_line);
            ~Dataframe();

            double get(int r, int c);
    };

    Dataframe::Dataframe()
    {
    }

    Dataframe::Dataframe(string file_path, int start_line){
        ifstream file(file_path);
        string line;
        int line_count = 0;

        vector<vector<double>> table;

        while (getline(file, line)) {
            line_count++;
            if(line_count >= start_line){
                vector<double> row;
                stringstream ls(line);
                string data;
                while (getline(ls, data, '\t')) {
                    row.push_back(stod(data));
                }
                table.push_back(row);
            }
        }

        this->table = table;
    }

    Dataframe::~Dataframe()
    {
    }

    double Dataframe::get(int r, int c){
        return table[r][c];
    }
}
