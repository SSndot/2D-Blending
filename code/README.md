# 使用说明：
1. 使用需要`OpenGL`与`Eigen`，然后构建工程加入文件
2. 程序分为读入给定多边形数据文件与用户自行输入多边形数据两种方式，默认是读入给定多边形数据文件方式。如果需要使用用户自行输入多边形数据方式，则需要将`Type.h`文件中的`UserInput`值改为1
3. 出于动画显示考虑，如果需要使用用户自行输入多边形数据方式，请将多边形坐标数据控制在`30*20`的范围内，如`canvas.png`图片显示
4. 读入给定多边形数据文件方式下不输出相关信息，用户自行输入多边形数据方式下输出相关信息，包括：多边形信息、匹配关系信息、仿射变换选取信息、中间帧多边形坐标信息
