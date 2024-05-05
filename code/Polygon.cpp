#define _USE_MATH_DEFINES
#include <cmath>

#include "Polygon.h"
#include "Func.h"
#include "Type.h"


void Polygon::AddVertex(const Point& vertex) {
    Triangle triangle;
    triangle.vertex = vertex;
    vertices.push_back(triangle);
}

double Polygon::DotProduct(const Point& v1, const Point& v2) const {
    return v1.x * v2.x + v1.y * v2.y;
}

double Polygon::VectorMagnitude(const Point& vec) const {
    return sqrt(vec.x * vec.x + vec.y * vec.y);
}

// ������p1Ϊ����ĸߵ�����
Point Polygon::CalculateHeightVector(const Point& p1, const Point& p2, const Point& p3) const {
    Point AC = { p1.x - p3.x, p1.y - p3.y };
    Point AB = { p1.x - p2.x, p1.y - p2.y };
    double AB_AB = DotProduct(AB, AB);
    double AC_AC = DotProduct(AC, AC);
    double AB_AC = DotProduct(AB, AC);
    double part1 = (AC_AC - AB_AC) / (AB_AB + AC_AC - 2 * AB_AC);
    double part2 = (AB_AB - AB_AC) / (AB_AB + AC_AC - 2 * AB_AC);
    return { part1 * AB.x + part2 * AB.x, part1 * AB.y + part2 * AB.y };
}

void Polygon::CalculateAngles() {
    size_t n = vertices.size();

    for (size_t i = 0; i < n; ++i) {
        const Point& p1 = vertices[(i + n - 1) % n].vertex;
        const Point& p2 = vertices[i].vertex;
        const Point& p3 = vertices[(i + 1) % n].vertex;

        vertices[i].angle = CalculateAngle(p1, p2, p3);
    }
}

void Polygon::CalculateSideLengths() {
    size_t n = vertices.size();

    for (size_t i = 0; i < n; ++i) {
        const Point& p1 = vertices[i].vertex;
        const Point& p2 = vertices[(i + n - 1) % n].vertex;
        const Point& p3 = vertices[(i + 1) % n].vertex;

        double length1 = CalculateDistance(p1, p2);
        double length2 = CalculateDistance(p1, p3);

        vertices[i].sideLengths = { length1, length2 };
    }
}

void Polygon::PrintPolygon() const {
    size_t n = vertices.size();
    for (size_t i = 0; i < n; ++i) {
        std::cout << "���� " << i + 1 << " (" << vertices[i].vertex.x << ", " << vertices[i].vertex.y << "):"
            << "  �Ƕȣ�" << vertices[i].angle
            << "  �߳���" << vertices[i].sideLengths.first << ", " << vertices[i].sideLengths.second << std::endl;
    }
}

double Polygon::CalculateDistance(const Point& p1, const Point& p2) const {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

// �������p2�����Ӧ�ĽǶ�
double Polygon::CalculateAngle(const Point& p1, const Point& p2, const Point& p3) const {
    double dx1 = p1.x - p2.x;
    double dy1 = p1.y - p2.y;
    double dx2 = p3.x - p2.x;
    double dy2 = p3.y - p2.y;

    double dotProduct = dx1 * dx2 + dy1 * dy2;
    double magnitude1 = sqrt(dx1 * dx1 + dy1 * dy1);
    double magnitude2 = sqrt(dx2 * dx2 + dy2 * dy2);

    double angle = acos(dotProduct / (magnitude1 * magnitude2));
    return angle * 180.0 / M_PI;
}

double Polygon::CalculateArea(const Point& p1, const Point& p2, const Point& p3) const {
    double area = 0.0;

    // ��������p1p2��p1p3�������ֵ
    double dx1 = p2.x - p1.x;
    double dy1 = p2.y - p1.y;
    double dx2 = p3.x - p1.x;
    double dy2 = p3.y - p1.y;

    // ������
    double crossProduct = dx1 * dy2 - dx2 * dy1;

    // ���������������ȡ����ֵ
    area = 0.5 * abs(crossProduct);

    return area;
}

// ����Shoelace Theorem�����������
double Polygon::CalculatePolygonArea() const {
    double area = 0.0;
    size_t numVertices = vertices.size();

    for (size_t i = 0; i < numVertices; ++i) {
        const Point& currentVertex = vertices[i].vertex;
        const Point& nextVertex = vertices[(i + 1) % numVertices].vertex;

        double crossProduct = currentVertex.x * nextVertex.y - nextVertex.x * currentVertex.y;
        area += crossProduct;
    }

    return std::abs(area / 2.0);
}

void Polygon::InputPolygon() {
    std::cout << "�밴���������εĶ�������(��(x,y)��ʽ����,��#����,���������붥��������Ķ����):" << std::endl;
    while (true) {
        std::string input;
        std::cin >> input;

        if (input == "#") {
            break;
        }

        try {
            Point point = parsePoint(input);
            AddVertex(point);
        }
        catch (const std::invalid_argument& e) {
            std::cout << "�Ƿ�����,�����������������: " << std::endl;
        }
    }
}

void Polygon::InputPolygon(const std::string fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << fileName << std::endl;
        return;
    }

    while (true) {
        std::string input;
        getline(file, input);

        if (input == "#") {
            break;
        }

        Point point = parsePoint(input);
        AddVertex(point);
    }
}

void Polygon::InitPolygon(const std::string name) {
#if UserInput
    InputPolygon();
#else
    InputPolygon(name);
#endif
    CalculateAngles();
    CalculateSideLengths();
}



double Polygon::CalculateSimilarity(const Triangle& t1, const Triangle& t2) const {
    double weight1 = 0.5, weight2 = 0.5; // w1, w2����Ȩ�ز���
    double nume = abs(t1.sideLengths.first * t2.sideLengths.second - t1.sideLengths.second * t2.sideLengths.first);
    double deno = t1.sideLengths.first * t2.sideLengths.second + t1.sideLengths.second * t2.sideLengths.first;
    double part1 = 1 - nume / deno;
    double part2 = 1 - (double)abs(t1.angle - t2.angle) / 360;
    return part1 * weight1 + part2 * weight2;
}

void Polygon::InitSimilarityMatrix(SimilarityMatrix& matrix, const Polygon& p) {
    // ȷ�������붥������ƥ��
    matrix.resize((vertices.size() >= p.vertices.size()) ? vertices.size() : p.vertices.size());
    for (auto& row : matrix) {
        row.resize((vertices.size() >= p.vertices.size()) ? p.vertices.size() : vertices.size());
    }

    // �������ƶȾ���
    auto it1 = (vertices.size() >= p.vertices.size()) ? vertices.begin() : p.vertices.begin();
    for (auto& row : matrix) {
        auto it2 = (vertices.size() >= p.vertices.size()) ? p.vertices.begin() : vertices.begin();
        for (double& similarity : row) {
            similarity = CalculateSimilarity(*it1, *it2);
            ++it2;
        }
        ++it1;
    }
}

void Polygon::SubSimilarityMatrix(SimilarityMatrix& matrix) {
    for (auto& row : matrix) {
        for (double& similarity : row) {
            similarity = 1 - similarity;
        }
    }
}

void Polygon::PrintSimilarityMatrix(const SimilarityMatrix& matrix) const {
    int width = 10;
    for (const auto& row : matrix) {
        for (const auto& similarity : row) {
            std::cout << std::setw(width) << similarity << " ";
        }
        std::cout << std::endl;
    }
}



double Polygon::dijkstra(const SimilarityMatrix& matrix, int start, std::vector<double>& distance, std::vector<int>& path) {
    distance.clear();
    path.clear();
    int v1, v2, v;
    double weight = 0.0, distTotal = 0.0;
    int rows = matrix.size();  // ��ȡ����
    int cols = (rows > 0) ? matrix[0].size() : 0;  // ��ȡ����

    // ��ʼ������Ϊ�����
    distance.resize(cols + 1, INF);
    distance[start] = matrix[1][start];

    // ���ȶ�������ѡ�������С�Ķ���
    std::priority_queue<pdv, std::vector<pdv>, std::greater<pdv>> pq;
    std::vector<int> stemp = { start };
    pq.push({ matrix[0][start], stemp });

    while (true) {
        // �洢·��
        std::vector<int> currPath = pq.top().second;
        path = currPath;
        // �õ���ǰ��
        int u = currPath.back();
        double dist = pq.top().first;
        distTotal = dist;
        pq.pop();

        // ·���нڵ��������������˳�
        if (path.size() == rows) {
            break;
        }

        // �������ж���
        for (int j = 0; j < cols; j++) {
            v1 = u;
            v2 = j;
            // ���·������ͨ�������ڽӶ���
            if (!IsMatrixPathConnected(start, v1, v2, cols)) continue;

            int newNum = (v1 == v2) ? 0 : 1;
            int matchedNum = CountDistinctElements(path);
            int remainedNum = rows - path.size() - 1;
            int targetNum = cols;

            if (IsMatrixPathConnected(start, v1, v2, cols) && !IsVertexNumRelated(newNum, matchedNum, remainedNum, targetNum)) continue;

            v = j;
            weight = matrix[path.size()][v];

            // ��������Ի�
            // ���ͨ����ǰ����u�����ڽӶ���v�ľ�����̣�����¾��벢�������ȶ���
            if (v == u || distTotal + weight < distance[v]) {
                std::vector<int> tempPath = currPath;
                tempPath.push_back(v);
                distance[v] = distTotal + weight;
                pq.push({ distance[v], tempPath });
            }
        }
    }

    return distTotal;
}

void Polygon::MatchPolygonPoints(const SimilarityMatrix& matrix, std::vector<pii>& matches) {
    matches.clear();
    int rows = matrix.size();
    int cols = (rows > 0) ? matrix[0].size() : 0;
    std::vector<double> distance;
    std::vector<double> len;
    std::vector<int> path;
    std::vector<std::vector<int>> paths;
    pii match;
    for (int i = 0; i < cols; i++) {
        len.push_back(dijkstra(matrix, i, distance, path));
        paths.push_back(path);
    }

    auto minElement = std::min_element(len.begin(), len.end());
    int index = std::distance(len.begin(), minElement);

    for (int i = 0; i < path.size(); i++) {
        
        
    }

    const std::vector<int>& row = paths[index]; // ��ȡ����
    // ʹ�õ�����������i�е�����Ԫ��
    for (int j = 0; j < row.size(); j++) {
        match.first = j;
        match.second = row[j];
        matches.push_back(match);
    }
}

void Polygon::PrintPolygonPointsMatches(const std::vector<pii>& matches) const {
    std::cout << "The matches are: " << std::endl;
    for (const auto& match : matches) {
        std::cout << "(" << match.first << ", " << match.second << ")" << std::endl;
    }
}

double Polygon::CalculateAffineSimilarity(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const {
    // eij��aij��i��ʾ�����Σ�j��ʾ����
    double e00 = CalculateDistance(vertices[p1.first].vertex, vertices[p2.first].vertex);
    double e01 = CalculateDistance(vertices[p2.first].vertex, vertices[p3.first].vertex);
    double e02 = CalculateDistance(vertices[p3.first].vertex, vertices[p1.first].vertex);
    double e10 = CalculateDistance(p.vertices[p1.second].vertex, p.vertices[p2.second].vertex);
    double e11 = CalculateDistance(p.vertices[p2.second].vertex, p.vertices[p3.second].vertex);
    double e12 = CalculateDistance(p.vertices[p3.second].vertex, p.vertices[p1.second].vertex);

    double a00 = CalculateAngle(vertices[p2.first].vertex, vertices[p1.first].vertex, vertices[p3.first].vertex);
    double a01 = CalculateAngle(vertices[p1.first].vertex, vertices[p2.first].vertex, vertices[p3.first].vertex);
    double a02 = CalculateAngle(vertices[p1.first].vertex, vertices[p3.first].vertex, vertices[p2.first].vertex);
    double a10 = CalculateAngle(p.vertices[p2.second].vertex, p.vertices[p1.second].vertex, p.vertices[p3.second].vertex);
    double a11 = CalculateAngle(p.vertices[p1.second].vertex, p.vertices[p2.second].vertex, p.vertices[p3.second].vertex);
    double a12 = CalculateAngle(p.vertices[p1.second].vertex, p.vertices[p3.second].vertex, p.vertices[p2.second].vertex);

    double nume = abs(e00 - e10) + abs(e01 - e11) + abs(e02 - e12);
    double deno = abs(e00 + e10) + abs(e01 + e11) + abs(e02 + e12);

    double anume = abs(a00 - a10) - abs(a01 - a11) - abs(a02 - a12);

    double part1 = 1 - nume / deno;
    double part2 = 1 - (double)anume / 360;

    double weight1 = 0.5, weight2 = 0.5; // Ȩ��ϵ��

    return part1 * weight1 + part2 * weight2;
}



double Polygon::CalculateAffineRotate(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const {
    Point h0 = CalculateHeightVector(vertices[p1.first].vertex, vertices[p2.first].vertex, vertices[p3.first].vertex);
    Point h1 = CalculateHeightVector(vertices[p1.second].vertex, vertices[p2.second].vertex, vertices[p3.second].vertex);

    double theta = acos(DotProduct(h0, h1) / (VectorMagnitude(h0) * VectorMagnitude(h1)));

    return 1 - (theta * 180.0 / M_PI) / 180;
}

double Polygon::CalculateAffineArea(const Polygon& p, const pii& p1, const pii& p2, const pii& p3) const {
    double area0 = CalculateArea(vertices[p1.first].vertex, vertices[p2.first].vertex, vertices[p3.first].vertex);
    double area1 = CalculateArea(vertices[p1.second].vertex, vertices[p2.second].vertex, vertices[p3.second].vertex);

    double parea0 = CalculatePolygonArea();
    double parea1 = p.CalculatePolygonArea();

    double A = (area0 + area1) / (parea0 + parea1);

    double d = 2; //����

    return A / (A + d);
}



void Polygon::SelectAffine(const Polygon& p, const std::vector<pii>& matches, std::vector<pii>& affine) {
    double w1 = 0.4, w2 = 0.4, w3 = 0.2;
    size_t n = vertices.size();
    size_t np = p.vertices.size();
    std::vector<pdv> smooths;
    int numMatches = matches.size();
    for (int i = 0; i < numMatches; ++i) {
        for (int j = i + 1; j < numMatches; ++j) {
            for (int k = j + 1; k < numMatches; ++k) {

                if (Point::ArePointsCoincide(p.vertices[matches[i].second].vertex, p.vertices[matches[j].second].vertex, p.vertices[matches[k].second].vertex))
                    continue;

                const pii& p11 = matches[i];
                const pii& p10 = { (matches[i].first + n - 1) % n, (matches[i].second + np - 1) % np };
                const pii& p12 = { (matches[i].first + 1) % n, (matches[i].second + 1) % np };

                const pii& p21 = matches[j];
                const pii& p20 = { (matches[j].first + n - 1) % n, (matches[j].second + np - 1) % np };
                const pii& p22 = { (matches[j].first + 1) % n, (matches[j].second + 1) % np };

                const pii& p31 = matches[k];
                const pii& p30 = { (matches[k].first + n - 1) % n, (matches[k].second + np - 1) % np };
                const pii& p32 = { (matches[k].first + 1) % n, (matches[k].second + 1) % np };

                double smooth1 = CalculateAffineSimilarity(p, p10, p11, p12) * w1 +
                    CalculateAffineRotate(p, p10, p11, p12) * w2 +
                    CalculateAffineArea(p, p10, p11, p12) * w3;

                double smooth2 = CalculateAffineSimilarity(p, p20, p21, p22) * w1 +
                    CalculateAffineRotate(p, p20, p21, p22) * w2 +
                    CalculateAffineArea(p, p20, p21, p22) * w3;

                double smooth3 = CalculateAffineSimilarity(p, p30, p31, p32) * w1 +
                    CalculateAffineRotate(p, p30, p31, p32) * w2 +
                    CalculateAffineArea(p, p30, p31, p32) * w3;

                smooths.push_back({ smooth1 * smooth2 * smooth3, {i, j, k} });
            }
        }
    }
    std::sort(smooths.begin(), smooths.end(), CompareSmoothElement);

    affine.clear();
    affine.resize(3);

    int i = 0;
    while (true) {
        //��Ҫ��֤ѡȡ�ķ���任���㲻����
        Point p1 = vertices[matches[smooths[i].second[0]].first].vertex;
        Point q1 = vertices[matches[smooths[i].second[1]].first].vertex;
        Point r1 = vertices[matches[smooths[i].second[2]].first].vertex;

        Point p2 = p.vertices[matches[smooths[i].second[0]].second].vertex;
        Point q2 = p.vertices[matches[smooths[i].second[1]].second].vertex;
        Point r2 = p.vertices[matches[smooths[i].second[2]].second].vertex;

        if (!IsCollinear(p1, q1, r1) && !IsCollinear(p2, q2, r2)) {
            affine[0] = matches[smooths[i].second[0]];
            affine[1] = matches[smooths[i].second[1]];
            affine[2] = matches[smooths[i].second[2]];
            break;
        }
        else
            i++;
    }
}

void Polygon::PrintAffine(const std::vector<pii>& affine) const {
    std::cout << "The affine is: " << std::endl;
    for (const auto& element : affine) {
        std::cout << "(" << element.first << ", " << element.second << ")" << std::endl;
    }
}



/* �������任���� */
void Polygon::GenerateAffineMatrix(const Polygon& p, const std::vector<pii>& affine, Eigen::Matrix2d& A, Eigen::Matrix<double, 1, 2>& T) {
    double _u0 = p.vertices[affine[0].second].vertex.x;
    double _u1 = p.vertices[affine[1].second].vertex.x;
    double _u2 = p.vertices[affine[2].second].vertex.x;

    double _v0 = p.vertices[affine[0].second].vertex.y;
    double _v1 = p.vertices[affine[1].second].vertex.y;
    double _v2 = p.vertices[affine[2].second].vertex.y;

    double u0 = vertices[affine[0].first].vertex.x;
    double u1 = vertices[affine[1].first].vertex.x;
    double u2 = vertices[affine[2].first].vertex.x;

    double v0 = vertices[affine[0].first].vertex.y;
    double v1 = vertices[affine[1].first].vertex.y;
    double v2 = vertices[affine[2].first].vertex.y;

    Eigen::Matrix<double, 3, 1> _u;
    Eigen::Matrix<double, 3, 1> _v;
    Eigen::Matrix<double, 3, 1> u;
    Eigen::Matrix<double, 3, 1> v;

    _u << _u0, _u1, _u2;
    _v << _v0, _v1, _v2;
    u << u0, u1, u2;
    v << v0, v1, v2;

    Eigen::Matrix<double, 3, 3> G;
    G << u(0), v(0), 1.0,
        u(1), v(1), 1.0,
        u(2), v(2), 1.0;

    Eigen::Matrix<double, 3, 1> b;
    b = _u;

    Eigen::Matrix<double, 3, 1> x = G.colPivHouseholderQr().solve(b);

    double a11 = x(0);
    double a21 = x(1);
    double a31 = x(2);

    b = _v;

    x = G.colPivHouseholderQr().solve(b);

    double a12 = x(0);
    double a22 = x(1);
    double a32 = x(2);

    A << a11, a12,
        a21, a22;

    T << a31, a32;
}

/* �������任��ֵ���� */
void Polygon::GenerateAffineInterpolationMatrix(const Eigen::Matrix2d& A, Eigen::Matrix2d& B, Eigen::Matrix2d& C) {
    double detA = A.determinant();
    double signDetA = (detA >= 0) ? 1.0 : -1.0;

    Eigen::Matrix2d M;
    M << A(1, 1), -A(1, 0),
        -A(0, 1), A(0, 0);

    B = A + signDetA * M;

    C = B.inverse() * A;
}

/* �������任��ֵ�� */
void Polygon::CalculateAffineInterpolation(const Eigen::Matrix2d& B, const Eigen::Matrix2d& C, const Eigen::Matrix<double, 1, 2>& T, const Point& p1, Point& p, double t) {
    double b11 = B(0, 0); // B(0, 0)
    double b12 = B(0, 1); // B(0, 1)
    double b21 = B(1, 0); // B(1, 0)
    double b22 = B(1, 1); // B(1, 1)

    double k = sqrt(B(0, 0) * B(0, 0) + B(1, 0) * B(1, 0));

    double theta = atan2(b21 / k, b11 / k);

    double ta = t * theta;
    double cos_a = std::cos(theta);
    double sin_a = std::sin(theta);
    double cos_ta = std::cos(ta);
    double sin_ta = std::sin(ta);

    double s11 = B(0, 0) / cos_a;
    double s12 = B(0, 1) / sin_a;
    double s21 = B(1, 0) / sin_a;
    double s22 = B(1, 1) / cos_a;

    Eigen::Matrix2d _B;
    _B << s11 * cos_ta, s12 * sin_ta,
          s21 * sin_ta, s22 * cos_ta;

    Eigen::Matrix2d M;
    M << 1, 0,
         0, 1;

    Eigen::Matrix2d _A;
    _A = (1 - t) * M + _B * (t * C);

    Eigen::Matrix<double, 1, 2> P1;
    P1 << p1.x, p1.y;

    Eigen::Matrix<double, 1, 2> P;

    P = P1 * _A + T * t;

    p.x = P(0);
    p.y = P(1);
}

/* �����м��������� */
void Polygon::GenerateIntermediatePoint(const std::vector<pii>& affine, const pii& vp, const Polygon& p, int frame, std::vector<Point>& inpoint) {
    Point pA1 = vertices[affine[0].first].vertex;
    Point pA2 = p.vertices[affine[0].second].vertex;
    Point pB1 = vertices[affine[1].first].vertex;
    Point pB2 = p.vertices[affine[1].second].vertex;
    Point pC1 = vertices[affine[2].first].vertex;
    Point pC2 = p.vertices[affine[2].second].vertex;
    Point pX1 = vertices[vp.first].vertex;
    Point pX2 = p.vertices[vp.second].vertex;

    Point u1 = StandardizeCoordinates(pA1, pB1, pC1, pX1);
    Point u2 = StandardizeCoordinates(pA2, pB2, pC2, pX2);

    Eigen::Matrix2d A, B, C;
    Eigen::Matrix<double, 1, 2> T;

    GenerateAffineMatrix(p, affine, A, T);
    GenerateAffineInterpolationMatrix(A, B, C);

    Point pA, pB, pC;

    inpoint.clear();
    inpoint.resize(0);

    double delta_t = 1 / (double)frame;

    for (int i = 0; i <= frame; i++) {
        double t = delta_t * i;
        double u = (1 - t) * u1.x + t * u2.x;
        double v = (1 - t) * u1.y + t * u2.y;

        CalculateAffineInterpolation(B, C, T, pA1, pA, t);
        CalculateAffineInterpolation(B, C, T, pB1, pB, t);
        CalculateAffineInterpolation(B, C, T, pC1, pC, t);

        Point X = { pB.x + u * (pA.x - pB.x) + v * (pC.x - pB.x), pB.y + u * (pA.y - pB.y) + v * (pC.y - pB.y) };
        inpoint.push_back(X);
    }
}

void Polygon::GenerateIntermediatePolygon(const std::vector<pii>& affine, const std::vector<pii>& matches, const Polygon& p, int frame, std::vector<pdp>& inpolygon) {
    for (int i = 0; i < vertices.size(); i++) {
        pii vp = { i, matches[i].second };
        std::vector<Point> inpoint;
        GenerateIntermediatePoint(affine, vp, p, frame, inpoint);
        
        inpolygon.push_back({ i, inpoint });
    }
}

void Polygon::PrintIntermediatePolygon(const std::vector<pdp>& inpolygon) const {
    for (const auto& pair : inpolygon) {
        std::cout << "First element of pair: " << pair.first << std::endl;
        std::cout << "Second element of pair: ";
        for (const auto& point : pair.second) {
            std::cout << "(" << point.x << ", " << point.y << ") ";
        }
        std::cout << std::endl;
    }
}