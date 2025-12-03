#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define EPSILON 0.01

void find_local_maximas(float *matrix, unsigned char *result, int startRow, int endRow, int startCol, int endCol, int M, int N);

void find_local_maximas(float *matrix, unsigned char *result, int startRow, int endRow, int startCol, int endCol, int M, int N)
{
    /*if (startRow >= endRow - 1 || startCol >= endCol - 1)
    {
        return;
    }

    int midRow = (startRow + endRow) / 2;
    int midCol = (startCol + endCol) / 2;

    // 查找左上象限的局部最大值
    find_local_maximas(matrix, result, startRow, midRow, startCol, midCol, M, N);
    // 查找右上象限的局部最大值
    find_local_maximas(matrix, result, startRow, midRow, midCol, endCol, M, N);
    // 查找左下象限的局部最大值
    find_local_maximas(matrix, result, midRow, endRow, startCol, midCol, M, N);
    // 查找右下象限的局部最大值
    find_local_maximas(matrix, result, midRow, endRow, midCol, endCol, M, N);

    // 处理交界处
    for (int i = startRow; i < midRow; ++i)
    {
        for (int j = startCol; j < midCol; ++j)
        {
            float value = matrix[(i * N) + j];
            if (fabs(value - 1.0) < EPSILON)
            {
                result[(i * N) + j] = 1; // 如果值接近1.0，直接标记为1
            }
            else if (value <= EPSILON)
            {
                result[(i * N) + j] = 0;
            }
            else
            {
                float top = (i == 0) ? -INFINITY : matrix[((i - 1) * N) + j];
                float bottom = (i == M - 1) ? -INFINITY : matrix[((i + 1) * N) + j];
                float left = (j == 0) ? -INFINITY : matrix[(i * N) + (j - 1)];
                float right = (j == N - 1) ? -INFINITY : matrix[(i * N) + (j + 1)];
                if (value > top && value > bottom && value > left && value > right)
                {
                    result[(i * N) + j] = 1;
                }
                else
                {
                    result[(i * N) + j] = 0;
                }
            }
        }
    }*/
    /*for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float value = matrix[(i * N) + j];
            if (value <= EPSILON) // fabs(value - 1.0) < EPSILON ||
            {
                result[(i * N) + j] = value <= EPSILON ? 0 : 1; // 如果值接近1.0，直接标记为1
                continue;
            }
            bool is_local_max = true;

            // 检查所有 8 个邻域
            for (int di = -1; di <= 1; ++di)
            {
                for (int dj = -1; dj <= 1; ++dj)
                {
                    if (di == 0 && dj == 0)
                        continue; // 跳过自身
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < M && nj >= 0 && nj < N)
                    {
                        if (value < matrix[(ni * N) + nj])
                        {
                            is_local_max = false;
                            break;
                        }
                    }
                }
                if (!is_local_max)
                    break;
            }

            result[(i * N) + j] = is_local_max ? 1 : 0;
        }
    }*/
    // #pragma omp parallel for collapse(2) // 并行化优化（可选）
    for (int i = 1; i < M - 1; ++i)
    {
        for (int j = 1; j < N - 1; ++j)
        {
            float val = matrix[i * N + j];
            // 手动展开邻域比较（8方向）
            result[i * N + j] = (val > EPSILON &&
                                 val >= matrix[(i - 1) * N + j - 1] &&
                                 val >= matrix[(i - 1) * N + j] &&
                                 val >= matrix[(i - 1) * N + j + 1] &&
                                 val >= matrix[i * N + j - 1] &&
                                 val >= matrix[i * N + j + 1] &&
                                 val >= matrix[(i + 1) * N + j - 1] &&
                                 val >= matrix[(i + 1) * N + j] &&
                                 val >= matrix[(i + 1) * N + j + 1])
                                    ? 1
                                    : 0;
        }
    }
}

// 外部接口函数
extern "C"
{
    void find_local_maximas_wrapper(float *matrix, unsigned char *result, int M, int N)
    {
        find_local_maximas(matrix, result, 0, M, 0, N, M, N);
    }
}

// 编译指令：g++ -O3 -march=native -funroll-loops -ffast-math  -shared -o find_local_maximas.dll find_local_maximas.cpp