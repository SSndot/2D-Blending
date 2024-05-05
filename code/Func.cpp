#include <Eigen/Dense>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <regex>

#include "Func.h"

int Orientation(const Point& p, const Point& q, const Point& r) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) {
        return 0;  // 共线
    }
    else if (val > 0) {
        return 1;  // 顺时针方向
    }
    else {
        return 2;  // 逆时针方向
    }
}

bool IsCollinear(const Point& p, const Point& q, const Point& r) {
    return Orientation(p, q, r) == 0;
}

bool CompareSmoothElement(const pdv& a, const pdv& b) {
    return a.first > b.first;
}

bool IsMatrixPathConnected(int sv, int v1, int v2, int targetTotal) {
    // 同一顶点之间存在路径
    if (v1 == v2)
        return true;
    else if (v2 == (v1 + 1) % targetTotal && v2 != sv)
        return true;
    else
        return false;
}

bool IsVertexNumRelated(int newNum, int matchedNum, int remainedNum, int targetNum) {
    return (newNum + matchedNum + remainedNum >= targetNum);
}

int CountDistinctElements(const std::vector<int>& v) {
    std::unordered_set<int> uniqueElements(v.begin(), v.end());
    return uniqueElements.size();
}

Point StandardizeCoordinates(const Point& A1, const Point& B1, const Point& C1, const Point& X1) {
    Eigen::Matrix2d A;
    Eigen::Vector2d b;

    A(0, 0) = A1.x - B1.x;
    A(0, 1) = C1.x - B1.x;
    A(1, 0) = A1.y - B1.y;
    A(1, 1) = C1.y - B1.y;

    b(0) = X1.x - B1.x;
    b(1) = X1.y - B1.y;

    Eigen::Vector2d solution = A.inverse() * b;

    Point result = { 0,0 };
    result.x = solution(0);
    result.y = solution(1);

    return result;
}

void WritePolygonDataToFile(const std::vector<pdp>& inpolygon, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile) {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }

    outFile << inpolygon.size() << std::endl;
    outFile << inpolygon[0].second.size() << std::endl;

    for (const auto& polygon : inpolygon) {
        outFile << polygon.first << std::endl;

        for (const auto& point : polygon.second) {
            outFile << "(" << point.x << ", " << point.y << ") ";
        }

        outFile << std::endl;
        outFile << std::endl;
    }

    outFile.close();
}

Point parsePoint(const std::string& input) {
    // 使用正则表达式匹配（x，y）格式，支持整数和浮点数
    std::regex pattern(R"(\((-?\d+(\.\d+)?),(-?\d+(\.\d+)?)\))");

    std::smatch matches;
    if (std::regex_search(input, matches, pattern)) {
        // 提取坐标字符串中的数字部分
        double x = std::stod(matches[1].str());
        double y = std::stod(matches[3].str());
        return { x, y };
    }
    else {
        // 非法输入
        throw std::invalid_argument("非法输入");
    }
}