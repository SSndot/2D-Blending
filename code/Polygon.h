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
    /*����γ�ʼ��*/
    void InitPolygon(const std::string fileName = " ");
    /*�������м��붥��*/
    void AddVertex(const Point& vertex);
    /*����Ƕ�*/
    void CalculateAngles();
    /*����߳�*/
    void CalculateSideLengths();
    /*��ӡ�������Ϣ*/
    void PrintPolygon() const;
    /*�������ƶȾ���*/
    void InitSimilarityMatrix(SimilarityMatrix& matrix, const Polygon& p);
    /*�������ƶȲ�����*/
    void SubSimilarityMatrix(SimilarityMatrix& matrix);
    /*��ӡ���ƶȾ���*/
    void PrintSimilarityMatrix(const SimilarityMatrix& matrix) const;
    /*����ζ���ƥ��*/
    void MatchPolygonPoints(const SimilarityMatrix& matrix, std::vector<pii>& matches);
    /*��ӡ����ζ���ƥ����*/
    void PrintPolygonPointsMatches(const std::vector<pii>& matches) const;
    /*ѡ�����任*/
    void SelectAffine(const Polygon& p, const std::vector<pii>& matches, std::vector<pii>& affine);
    /*��ӡ����任ѡ����*/
    void PrintAffine(const std::vector<pii>& affine) const;
    /*�����м�֡�����*/
    void GenerateIntermediatePolygon(const std::vector<pii>& affine, const std::vector<pii>& matches, const Polygon& p, int frame, std::vector<pdp>& inpolygon);
    /*��ӡ�м�֡���������*/
    void PrintIntermediatePolygon(const std::vector<pdp>& inpolygon) const;

private:
    /*�����ʼ���������*/
    void InputPolygon();
    void InputPolygon(const std::string name);
    /*������*/
    double DotProduct(const Point& vec1, const Point& vec2) const;
    /*��������ģ��*/
    double VectorMagnitude(const Point& vec) const;
    /*���������*/
    Point CalculateHeightVector(const Point& p1, const Point& p2, const Point& p3) const;
    /*�������ƶ�*/
    double CalculateSimilarity(const Triangle& t1, const Triangle& t2) const;
    /*���㶥�����*/
    double CalculateDistance(const Point& p1, const Point& p2) const;
    /*����Ƕ�*/
    double CalculateAngle(const Point& p1, const Point& p2, const Point& p3) const;
    /*���㶥��Χ�����*/
    double CalculateArea(const Point& p1, const Point& p2, const Point& p3) const;
    /*�����������*/
    double CalculatePolygonArea() const;
    /*����ƥ���㷨*/
    double dijkstra(const SimilarityMatrix& matrix, int start, std::vector<double>& distance, std::vector<int>& path);
    /*�������任ѡ�����ƶ�����*/
    double CalculateAffineSimilarity(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const;
    /*�������任ѡ����ת������*/
    double CalculateAffineRotate(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const;
    /*�������任ѡ���������*/
    double CalculateAffineArea(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const;
    /*���ɷ���任����A��T*/
    void GenerateAffineMatrix(const Polygon& p, const std::vector<pii>& affine, Eigen::Matrix2d& A, Eigen::Matrix<double, 1, 2>& T);
    /*���ɷ���任����A�ķֽ����B��C*/
    void GenerateAffineInterpolationMatrix(const Eigen::Matrix2d& A, Eigen::Matrix2d& B, Eigen::Matrix2d& C);
    /*���ɷ���任��ֵ����*/
    void CalculateAffineInterpolation(const Eigen::Matrix2d& B, const Eigen::Matrix2d& C, const Eigen::Matrix<double, 1, 2>& T, const Point& p1, Point& p, double t);
    /*�����м��*/
    void GenerateIntermediatePoint(const std::vector<pii>& affine, const pii& vp, const Polygon& p, int frame, std::vector<Point>& inpoint);
};