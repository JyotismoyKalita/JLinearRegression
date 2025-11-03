#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "j_numframe.h"

float square(float x)
{
    return x * x;
}

float cost_function(numframe *x, numframe *y, float w, float b)
{
    float cost = 0;
    int n = y->rows;
    for (int i = 1; i < n; i++)
    {
        float actual = w * x->arr[i] + b;
        float expected = y->arr[i];
        cost += square(actual - expected);
    }
    return cost / 2 * (float)n;
}

void gradient(float *dw, float *db, numframe *x, numframe *y, float w, float b)
{
    float t_dw = 0, t_db = 0;
    int n = y->rows;
    for (int i = 0; i < n; i++)
    {
        float inner = w * x->arr[i] + b;
        float outer = inner - y->arr[i];
        t_dw += outer * x->arr[i];
        t_db += outer;
    }
    *dw = t_dw / (float)n;
    *db = t_db / (float)n;
}

void gradient_decent(numframe *x, numframe *y, float alpha, int iterations, float minimum_cost, bool verbose)
{
    if (x == NULL || y == NULL)
        return;
    float w = 0;
    float b = 0;
    for (int i = 0; i < iterations; i++)
    {
        float dw, db;
        gradient(&dw, &db, x, y, w, b);
        w = w - alpha * dw;
        b = b - alpha * db;
        float cost = cost_function(x, y, w, b);
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
    numframe *x_train = nframe_split_col(0, 0, 1, ndat);
    nframe_show(x_train);
    numframe *y_train = nframe_split_col(1, 1, 0, ndat);
    nframe_show(y_train);
    gradient_decent(x_train, y_train, 0.01, 10000, 0.0001, 0);
    nframe_destroy(ndat);
    nframe_destroy(x_train);
    nframe_destroy(y_train);
    return 0;
}