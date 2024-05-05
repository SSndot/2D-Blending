#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

#include "Type.h"

const char* title = "2DShapeBlending";

int windowWidth = 1800;
int windowHeight = 1200;

double scaleX = 45.0;
double scaleY = 30.0;

std::vector<std::vector<Point>> polygons; // 存储多边形顶点坐标的向量

void readPolygonData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }

    polygons.clear();

    int polygonCount, vertexCount;
    file >> vertexCount >> polygonCount;

    for (int i = 0; i < vertexCount; i++) {
        int vertexID;
        file >> vertexID;

        std::vector<Point> vertices;
        vertices.resize(vertexCount);
        polygons.resize(polygonCount);


        for (int j = 0; j < polygonCount; j++) {
            char dummy;
            Point vertex;
            file >> dummy >> vertex.x >> dummy >> vertex.y >> dummy;
            vertex.x /= scaleX;
            vertex.y /= scaleY;
            polygons[j].push_back(vertex);
        }

    }

    file.close();
}

void renderPolygon(const std::vector<Point>& polygon) {
    // 创建顶点数组和颜色数组
    std::vector<float> vertices;

    // 计算顶点个数
    int vertexCount = polygon.size();

    for (int i = 0; i < vertexCount; i++) {
        const auto& vertex = polygon[i];
        vertices.push_back(vertex.x);
        vertices.push_back(vertex.y);
    }

    // 创建顶点缓冲对象（VBO）和颜色缓冲对象（CBO）
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    // 创建顶点数组对象（VAO）
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // 绑定顶点缓冲对象（VBO）
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // 绘制多边形
    glDrawArrays(GL_LINE_LOOP, 0, vertexCount);

    // 清理
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}



void render() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    static std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = (currentTime - startTime);

    int polygonIndex = static_cast<int>(elapsedSeconds.count() / 0.02) % polygons.size();
    const std::vector<Point>& polygon = polygons[polygonIndex];

    renderPolygon(polygon);
}

void reshape(GLFWwindow* window, int width, int height) {
    // 更新窗口宽度和高度
    windowWidth = width;
    windowHeight = height;

    // 设置视口
    glViewport(0, 0, width, height);
}

int AnimationInit() {
    // 初始化GLFW
    if (!glfwInit())
        return -1;

    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, title, NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    // 设置当前上下文
    glfwMakeContextCurrent(window);

    // 初始化GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        return -1;
    }

    // 设置视口
    glViewport(0, 0, windowWidth, windowHeight);

    // 设置窗口大小变化回调函数
    glfwSetFramebufferSizeCallback(window, reshape);

    // 读取多边形数据
    readPolygonData("PolygonData.txt");

    while (!glfwWindowShouldClose(window)) {
        // 渲染
        render();

        // 交换缓冲区和轮询事件
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 终止GLFW
    glfwTerminate();

    return 0;
}