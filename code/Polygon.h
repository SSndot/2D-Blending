#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <regex>
#include <cmath>
#include <queue>
#include <limits>
#include <unordered_set>
#include <Eigen/Dense>
#include <fstream>

#include "Type.h"



class Polygon {
private:
    std::vector<Triangle> vertices;

public:
    /*多边形初始化*/
    void InitPolygon(const std::string fileName = " ");
    /*向多边形中加入顶点*/
    void AddVertex(const Point& vertex);
    /*计算角度*/
    void CalculateAngles();
    /*计算边长*/
    void CalculateSideLengths();
    /*打印多边形信息*/
    void PrintPolygon() const;
    /*生成相似度矩阵*/
    void InitSimilarityMatrix(SimilarityMatrix& matrix, const Polygon& p);
    /*生成相似度补矩阵*/
    void SubSimilarityMatrix(SimilarityMatrix& matrix);
    /*打印相似度矩阵*/
    void PrintSimilarityMatrix(const SimilarityMatrix& matrix) const;
    /*多边形顶点匹配*/
    void MatchPolygonPoints(const SimilarityMatrix& matrix, std::vector<pii>& matches);
    /*打印多边形顶点匹配结果*/
    void PrintPolygonPointsMatches(const std::vector<pii>& matches) const;
    /*选择仿射变换*/
    void SelectAffine(const Polygon& p, const std::vector<pii>& matches, std::vector<pii>& affine);
    /*打印仿射变换选择结果*/
    void PrintAffine(const std::vector<pii>& affine) const;
    /*生成中间帧多边形*/
    void GenerateIntermediatePolygon(const std::vector<pii>& affine, const std::vector<pii>& matches, const Polygon& p, int frame, std::vector<pdp>& inpolygon);
    /*打印中间帧多边形数据*/
    void PrintIntermediatePolygon(const std::vector<pdp>& inpolygon) const;

private:
    /*输入初始多边形数据*/
    void InputPolygon();
    void InputPolygon(const std::string name);
    /*计算点乘*/
    double DotProduct(const Point& vec1, const Point& vec2) const;
    /*计算向量模长*/
    double VectorMagnitude(const Point& vec) const;
    /*计算高向量*/
    Point CalculateHeightVector(const Point& p1, const Point& p2, const Point& p3) const;
    /*计算相似度*/
    double CalculateSimilarity(const Triangle& t1, const Triangle& t2) const;
    /*计算顶点距离*/
    double CalculateDistance(const Point& p1, const Point& p2) const;
    /*计算角度*/
    double CalculateAngle(const Point& p1, const Point& p2, const Point& p3) const;
    /*计算顶点围成面积*/
    double CalculateArea(const Point& p1, const Point& p2, const Point& p3) const;
    /*计算多边形面积*/
    double CalculatePolygonArea() const;
    /*顶点匹配算法*/
    double dijkstra(const SimilarityMatrix& matrix, int start, std::vector<double>& distance, std::vector<int>& path);
    /*计算仿射变换选择相似度因子*/
    double CalculateAffineSimilarity(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const;
    /*计算仿射变换选择旋转角因子*/
    double CalculateAffineRotate(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const;
    /*计算仿射变换选择面积因子*/
    double CalculateAffineArea(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const;
    /*生成仿射变换矩阵A、T*/
    void GenerateAffineMatrix(const Polygon& p, const std::vector<pii>& affine, Eigen::Matrix2d& A, Eigen::Matrix<double, 1, 2>& T);
    /*生成仿射变换矩阵A的分解矩阵B、C*/
    void GenerateAffineInterpolationMatrix(const Eigen::Matrix2d& A, Eigen::Matrix2d& B, Eigen::Matrix2d& C);
    /*生成仿射变换插值矩阵*/
    void CalculateAffineInterpolation(const Eigen::Matrix2d& B, const Eigen::Matrix2d& C, const Eigen::Matrix<double, 1, 2>& T, const Point& p1, Point& p, double t);
    /*计算中间点*/
    void GenerateIntermediatePoint(const std::vector<pii>& affine, const pii& vp, const Polygon& p, int frame, std::vector<Point>& inpoint);
};