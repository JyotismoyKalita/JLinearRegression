#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "j_lr.h"

typedef struct lr_mdl
{
    float b;
    int count;
    float w[];
} lr_model;

float square(float x)
{
    return x * x;
}

void lr_train_test_split(numframe **x_train, numframe **y_train, numframe **x_test, numframe **y_test, numframe *original, const float ratio, const bool shuffle, const int seed, const int start, const int end, const int y_index)
{
    numframe *split1, *split2;
    nframe_split_row(ratio, original, &split1, &split2, shuffle, seed);
    *x_train = nframe_split_col(start, end, y_index, split1);
    *y_train = nframe_split_col(y_index, y_index, 0, split1);
    *x_test = nframe_split_col(start, end, y_index, split2);
    *y_test = nframe_split_col(y_index, y_index, 0, split2);
}

float cost_function(const numframe *x, const numframe *y, const float *w, const float b)
{
    float cost = 0;
    int n = y->rows;
    int cols = x->cols;
    for (int i = 0; i < n; i++)
    {
        float actual = b;
        for (int j = 0; j < cols; j++)
        {
            actual += w[j] * x->arr[i * cols + j];
        }
        float expected = y->arr[i];
        cost += square(actual - expected);
    }
    return cost / 2 * (float)n;
}

void gradient(float *dw, float *db, const numframe *x, const numframe *y, const float *w, const float b)
{
    int n = y->rows;
    int cols = x->cols;
    float t_dw[cols], t_db = 0;
    memset(t_dw, 0, cols * sizeof(t_dw[0]));
    for (int i = 0; i < n; i++)
    {
        float inner = b;
        for (int j = 0; j < cols; j++)
        {
            inner += w[j] * x->arr[i * cols + j];
        }
        float outer = inner - y->arr[i];
        for (int j = 0; j < cols; j++)
        {
            t_dw[j] += outer * x->arr[i * cols + j];
        }
        t_db += outer;
    }
    for (int j = 0; j < cols; j++)
    {
        dw[j] = t_dw[j] / (float)n;
    };
    *db = t_db / (float)n;
}

void gradient_decent(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model)
{
    if (x == NULL || y == NULL)
        return;
    int cols = x->cols;
    float w[cols];
    for (int i = 0; i < cols; i++)
        w[i] = model->w[i];
    float b = model->b, cost;
    for (int i = 0; i < iter; i++)
    {
        float dw[cols], db = 0;
        memset(dw, 0, cols * sizeof(dw[0]));
        gradient(dw, &db, x, y, w, b);
        for (int j = 0; j < cols; j++)
        {
            w[j] = w[j] - alpha * dw[j];
        }
        b = b - alpha * db;
        cost = cost_function(x, y, w, b);
        if (verbose == 1)
            printf("Cost at Iteration %d: %f\n", i, cost);
        if (cost < minimum_cost)
            break;
    }
    printf("Minimum MSE: %f\n", cost);
    printf("Final Function: %s = ", y->features[0]);
    model->b = b;
    for (int i = 0; i < cols; i++)
    {
        float w_val = w[i];
        model->w[i] = w_val;
        printf("%.2f * %s + ", w_val, x->features[i]);
    }
    printf("%.2f\n", b);
}

lr_model *lr_model_fit(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose)
{
    int model_size = sizeof(lr_model) + x->cols * sizeof(float);
    lr_model *model = (lr_model *)malloc(model_size);
    memset(model, 0, model_size);
    model->count = x->cols;
    gradient_decent(x, y, alpha, iter, minimum_cost, verbose, model);
    return model;
}

void *lr_model_retrain(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model)
{
    gradient_decent(x, y, alpha, iter, minimum_cost, verbose, model);
}

numframe *lr_model_predict(const numframe *x, const lr_model *model, const char *header)
{
    if (x == NULL)
        return NULL;
    numframe *ndat = (numframe *)malloc(sizeof(numframe));
    ndat->cols = 1;
    int rows = x->rows;
    int cols = x->cols;
    ndat->rows = rows;
    ndat->features = (char **)malloc(sizeof(char *));
    ndat->features[0] = strdup(header);
    ndat->arr = (float *)malloc(sizeof(float) * rows);
    for (int i = 0; i < rows; i++)
    {
        ndat->arr[i] = model->b;
        for (int j = 0; j < cols; j++)
        {
            ndat->arr[i] += model->w[j] * x->arr[i * cols + j];
        }
    }
    return ndat;
}

float lr_calculate_mse(const numframe *predict, const numframe *y)
{
    if (predict == NULL || y == NULL)
        return 0.00;
    float mse = 0;
    int n = predict->rows;
    for (int i = 0; i < n; i++)
    {
        mse += square(predict->arr[i] - y->arr[i]);
    }
    return mse / (float)n;
}

void lr_model_save(const char *filename, const lr_model *model)
{
    FILE *mf = fopen(filename, "wb");
    if (mf == NULL)
    {
        printf("Error saving...\n");
        return;
    }
    fwrite(&model->b, sizeof(float), 1, mf);
    fwrite(&model->count, sizeof(int), 1, mf);
    fwrite(&model->w, sizeof(float), model->count, mf);

    fclose(mf);
}

lr_model *lr_model_load(const char *filename)
{
    FILE *mf = fopen(filename, "rb");
    if (mf == NULL)
    {
        printf("Error loading...\n");
        return NULL;
    }
    float b;
    int count;
    fread(&b, sizeof(float), 1, mf);
    fread(&count, sizeof(int), 1, mf);
    int size = sizeof(lr_model) + sizeof(float) * count;
    lr_model *model = (lr_model *)malloc(size);
    model->b = b;
    model->count = count;
    fread(&model->w, sizeof(float), count, mf);
    fclose(mf);
    return model;
}

void lr_model_destroy(lr_model *model)
{
    free(model);
}