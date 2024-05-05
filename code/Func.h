#pragma once
#include <string>
#include "Type.h"

int Orientation(const Point& p, const Point& q, const Point& r);
bool IsCollinear(const Point& p, const Point& q, const Point& r);
bool IsMatrixPathConnected(int sv, int v1, int v2, int targetTotal);
bool IsVertexNumRelated(int newNum, int matchedNum, int remainedNum, int targetNum);
bool CompareSmoothElement(const pdv& a, const pdv& b);
int CountDistinctElements(const std::vector<int>& v);
void WritePolygonDataToFile(const std::vector<pdp>& inpolygon, const std::string& filename);
Point StandardizeCoordinates(const Point& A1, const Point& B1, const Point& C1, const Point& X1);
Point parsePoint(const std::string& input);