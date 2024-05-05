#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <queue>
#include <limits>
#include <Eigen/Dense>
#include <fstream>

#include "Polygon.h"
#include "Func.h"
#include "CreateAnimation.h"

std::vector<std::string> fileName = {
    "InputData/angel_1.txt",
    "InputData/angel_2.txt"
};

int main() {
    Polygon polygon1;
    Polygon polygon2;

#if UserInput
    polygon1.InitPolygon();
    polygon2.InitPolygon();
#else
    polygon1.InitPolygon(fileName[0]);
    polygon2.InitPolygon(fileName[1]);
#endif

#if UserInput
    std::cout << "第一个多边形的数据：" << std::endl;
    polygon1.PrintPolygon();

    std::cout << "第二个多边形的数据：" << std::endl;
    polygon2.PrintPolygon();
#endif

    SimilarityMatrix matrix;
    polygon1.InitSimilarityMatrix(matrix, polygon2);
    polygon1.SubSimilarityMatrix(matrix);

#if UserInput
    polygon1.PrintSimilarityMatrix(matrix);
#endif

    std::vector<pii> matches;
    polygon1.MatchPolygonPoints(matrix, matches);

#if UserInput
    polygon1.PrintPolygonPointsMatches(matches);
#endif

    std::vector<pii> affine;
    polygon1.SelectAffine(polygon2, matches, affine);

#if UserInput
    polygon1.PrintAffine(affine);
#endif

    std::vector<pdp> inpolygon;
    int frame = 120;

    polygon1.GenerateIntermediatePolygon(affine, matches, polygon2, frame, inpolygon);

#if UserInput
    polygon1.PrintIntermediatePolygon(inpolygon);
#endif

    const std::string filename = "PolygonData.txt";
    // 打开文件并清空内容
    try {
        std::ofstream file(filename, std::ios::trunc);
        if (file.is_open()) {
            file.close();
        }
        else {
            throw std::runtime_error("Can't open the file!");
        }
    }
    catch (const std::exception& e) {
        std::cout << "error：" << e.what() << std::endl;
    }
        
    WritePolygonDataToFile(inpolygon, filename);

    AnimationInit();

    return 0;
}