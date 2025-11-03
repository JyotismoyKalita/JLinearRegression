#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "j_numframe.h"

float cost_function(numframe *ndat, int x_index, int y_index, float m, float b)
{
    float cost = 0;
    int n = ndat->rows;
    int cols = ndat->cols;
    for (int i = 1; i < n; i++)
    {
        float actual = m * ndat->arr[i * cols + x_index] + b;
        float expected = ndat->arr[i * cols + y_index];
        cost += square(actual - expected);
    }
    return cost / 2 * (float)n;
}

float dw(numframe *ndat, int x_index, int y_index, float m, float b)
{
    float p_dw = 0;
    int n = ndat->rows;
    int cols = ndat->cols;
    for (int i = 0; i < n; i++)
    {
        float inner = m * ndat->arr[i * cols + x_index] + b;
        float outer = ndat->arr[i * cols + x_index] * (inner - ndat->arr[i * cols + y_index]);
        p_dw += outer;
    }
    return p_dw / (float)n;
}

float db(numframe *ndat, int x_index, int y_index, float m, float b)
{
    float p_db = 0;
    int n = ndat->rows;
    int cols = ndat->cols;
    for (int i = 0; i < n; i++)
    {
        float inner = m * ndat->arr[i * cols + x_index] + b;
        float outer = inner - ndat->arr[i * cols + y_index];
        p_db += outer;
    }
    return p_db / (float)n;
}

void gradient_decent(numframe *ndat, int x_index, int y_index, float alpha, int iterations, float minimum_cost, bool verbose)
{
    if (ndat == NULL)
        return;
    float w = 0;
    float b = 0;
    for (int i = 0; i < iterations; i++)
    {
        w = w - alpha * dw(ndat, x_index, y_index, w, b);
        b = b - alpha * db(ndat, x_index, y_index, w, b);
        float cost = cost_function(ndat, x_index, y_index, w, b);
        if (verbose == 1)
            printf("Cost at Iteration %d: %f\n", i, cost);
        if (cost < minimum_cost)
            break;
    }
    printf("Final Function: y = %.2f * x + %.2f\n", w, b);
}

int main()
{
    numframe *ndat = nframe_readcsv("data.csv", ',', 128);
    nframe_show(ndat);
    numframe *ndat1, *ndat2;
    nframe_split(0.7, ndat, &ndat1, &ndat2, 1, 42);
    nframe_show(ndat1);
    nframe_show(ndat2);
    gradient_decent(ndat1, 0, 1, 0.01, 10000, 0.0001, 0);
    nframe_destroy(ndat);
    nframe_destroy(ndat1);
    nframe_destroy(ndat2);
    return 0;
}