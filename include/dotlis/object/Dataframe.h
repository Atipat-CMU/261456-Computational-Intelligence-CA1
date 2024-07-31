#ifndef DATAFRAME_H
#define DATAFRAME_H

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
            int width;

        public:
            Dataframe();
            Dataframe(const string& file_path, int start_line);
            Dataframe(vector<vector<double>> &table);
            ~Dataframe();

            int get_width() const;
            int get_depth() const;
            double get(int r, int c) const;
            vector<double> getRow(int r) const;

            Dataframe get_column(const vector<int>& columns) const;
            Dataframe get_column_without(const vector<int>& columns) const;
    };

    Dataframe::Dataframe()
    {
    }

    Dataframe::Dataframe(const string& file_path, int start_line){
        ifstream file(file_path);
        if (!file.is_open()) {
            throw runtime_error("Could not open file");
        }

        string line;
        int line_count = 0;

        vector<vector<double>> table;

        while (getline(file, line)) {
            line_count++;
            if(line_count >= start_line){
                vector<double> row;
                stringstream ls(line);
                string data;
                int count = 0;
                while (getline(ls, data, '\t')) {
                    row.push_back(stod(data));
                    count++;
                }
                width = count;
                table.push_back(row);
            }
        }
        file.close();
        this->table = table;
    }

    Dataframe::Dataframe(vector<vector<double>> &table){
        this->table = table;
        if (!table.empty()) {
            width = table[0].size();
        } else {
            width = 0;
        }
    }

    Dataframe::~Dataframe()
    {
    }

    int Dataframe::get_width() const{
        return width;
    }

    int Dataframe::get_depth() const{
        return table.size();
    }

    double Dataframe::get(int r, int c) const{
        if (r >= get_depth() || c >= get_width()) {
            throw out_of_range("Index out of range");
        }
        return table[r][c];
    }

    vector<double> Dataframe::getRow(int r) const{
        return table[r];
    }

    Dataframe Dataframe::get_column(const vector<int>& columns) const{
        vector<vector<double>> new_table;
        for(int r = 0; r < this->get_depth(); r++){
            vector<double> row;
            for(int c : columns){
                row.push_back(this->get(r, c));
            }
            new_table.push_back(row);
        }
        return Dataframe(new_table);
    }

    Dataframe Dataframe::get_column_without(const vector<int>& columns) const{
        vector<int> selected;
        for(int i = 0; i < this->width; i++){
            bool is_needed = true;
            for(int c : columns){
                if(i == c){
                    is_needed = false;
                    break;
                }
            }
            if(is_needed) selected.push_back(i);
        }
        return get_column(selected);
    }
}


#endif
