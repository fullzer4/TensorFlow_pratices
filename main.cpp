#include "ETL.h"

#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

int main(int argc, char *argv[]){

    ETL etl(agrv[1], argv[2], argv[3]);

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    return EXIT_SUCCESS;
}