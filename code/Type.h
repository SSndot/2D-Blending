#pragma once
#pragma execution_character_set("utf-8")  

#include <vector>
#include <iostream>

#define UserInput 0
#define EV -1
#define INF std::numeric_limits<int>::max()

// 定义点
struct Point {
    double x;
    double y;
    static bool ArePointsEqual(const Point& p1, const Point& p2);
    static bool ArePointsCoincide(const Point& p1, const Point& p2, const Point& p3);
};

// 定义顶点
struct Triangle {
    Point vertex;
    double angle;
    std::pair<double, double> sideLengths;
};

using SimilarityMatrix = std::vector<std::vector<double>>;
using pii = std::pair<int, int>;
using pdi = std::pair<double, int>;
using pdd = std::pair<double, double>;
using pdv = std::pair<double, std::vector<int>>;
using pdp = std::pair<int, std::vector<Point>>;