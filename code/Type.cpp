#include <Eigen/Dense>
#include "Type.h"

bool Point::ArePointsEqual(const Point& p1, const Point& p2) {
    return (p1.x == p2.x && p1.y == p2.y);
}

bool Point::ArePointsCoincide(const Point& p1, const Point& p2, const Point& p3) {
    return (ArePointsEqual(p1, p2) || ArePointsEqual(p2, p3) || ArePointsEqual(p1, p3));
}

