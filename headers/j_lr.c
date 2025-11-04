#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "j_lr.h"

typedef struct lr_mdl
{
    float b;
    int count;
    unsigned char normalize;
    float w[];
} lr_model;

float square(float x)
{
    return x * x;
}

numframe *lr_normalize_features(const numframe *x)
{
    numframe *n_x = (numframe *)malloc(sizeof(numframe));
    int rows = x->rows;
    int cols = x->cols;
    int size = rows * cols;
    n_x->rows = rows;
    n_x->cols = cols;
    n_x->features = (char **)malloc(sizeof(char *) * cols);
    for (int i = 0; i < cols; i++)
    {
        n_x->features[i] = strdup(x->features[i]);
    }
    n_x->arr = (float *)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++)
    {
        n_x->arr[i] = x->arr[i];
    }

    for (int j = 0; j < cols; j++)
    {
        float sum = 0, sumSq = 0;
        for (int i = 0; i < rows; i++)
        {
            float val = n_x->arr[i * cols + j];
            sum += val;
            sumSq += val * val;
        }
        float mean = sum / rows;
        float variance = (sumSq / rows) - (mean * mean);
        float std = sqrtf(variance);
        if (std == 0)
            std = 1;
        for (int i = 0; i < rows; i++)
        {
            n_x->arr[i * cols + j] = (n_x->arr[i * cols + j] - mean) / std;
        }
    }
    return n_x;
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
    return cost / (2 * (float)n);
}

void gradient(float *dw, float *db, const numframe *x, const numframe *y, const float *w, const float b)
{
    int n = y->rows;
    int cols = x->cols;
    float *t_dw, t_db = 0;
    t_dw = calloc(cols, sizeof(float));
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
    free(t_dw);
}

void gradient_decent(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model)
{
    if (x == NULL || y == NULL)
        return;
    numframe *n_x;
    if (model->normalize == 1)
        n_x = lr_normalize_features(x);
    else
        n_x = x;
    int cols = n_x->cols;
    float *w;
    w = (float *)malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; i++)
        w[i] = model->w[i];
    float b = model->b, cost;
    for (int i = 0; i < iter; i++)
    {
        float *dw, db = 0;
        dw = calloc(cols, sizeof(float));
        gradient(dw, &db, n_x, y, w, b);
        for (int j = 0; j < cols; j++)
        {
            w[j] = w[j] - alpha * dw[j];
        }
        b = b - alpha * db;
        cost = cost_function(n_x, y, w, b);
        free(dw);
        if (verbose > 2)
            printf("Cost at Iteration %d: %f\n", i, cost);
        if (cost < minimum_cost)
            break;
    }
    if (verbose > 0)
        printf("Train MSE: %f\n", cost);
    if (verbose > 1)
        printf("Final Function: %s = ", y->features[0]);
    model->b = b;
    for (int i = 0; i < cols; i++)
    {
        float w_val = w[i];
        model->w[i] = w_val;
        if (verbose > 1)
            printf("%f * %s + ", w_val, n_x->features[i]);
    }
    if (verbose > 1)
        printf("%.2f\n", b);
    free(w);
    if (model->normalize == 1)
        nframe_destroy(n_x);
}

lr_model *lr_model_fit(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, const bool normalize)
{
    int model_size = sizeof(lr_model) + x->cols * sizeof(float);
    lr_model *model = (lr_model *)malloc(model_size);
    memset(model, 0, model_size);
    model->count = x->cols;
    model->normalize = normalize;
    gradient_decent(x, y, alpha, iter, minimum_cost, verbose, model);
    return model;
}

void *lr_model_retrain(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model)
{
    gradient_decent(x, y, alpha, iter, minimum_cost, verbose, model);
}

numframe *lr_model_predict(numframe *x, const lr_model *model, const char *header)
{
    if (x == NULL)
        return NULL;
    numframe *n_x;
    if (model->normalize == 1)
        n_x = lr_normalize_features(x);
    else
        n_x = x;

    numframe *ndat = (numframe *)malloc(sizeof(numframe));
    ndat->cols = 1;
    int rows = n_x->rows;
    int cols = n_x->cols;
    ndat->rows = rows;
    ndat->features = (char **)malloc(sizeof(char *));
    ndat->features[0] = strdup(header);
    ndat->arr = (float *)malloc(sizeof(float) * rows);
    for (int i = 0; i < rows; i++)
    {
        ndat->arr[i] = model->b;
        for (int j = 0; j < cols; j++)
        {
            ndat->arr[i] += model->w[j] * n_x->arr[i * cols + j];
        }
    }
    if (model->normalize == 1)
        nframe_destroy(n_x);
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
    fwrite(&model->normalize, sizeof(unsigned char), 1, mf);
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
    bool normalize;
    fread(&b, sizeof(float), 1, mf);
    fread(&count, sizeof(int), 1, mf);
    fread(&normalize, sizeof(bool), 1, mf);
    int size = sizeof(lr_model) + sizeof(float) * count;
    lr_model *model = (lr_model *)malloc(size);
    model->b = b;
    model->count = count;
    model->normalize = normalize;
    fread(&model->w, sizeof(float), count, mf);
    fclose(mf);
    return model;
}

void *lr_model_show_coeff(const lr_model *model, char **feature_list)
{
    for (int i = 0; i < model->count; i++)
    {
        printf("%s\t%f\n", feature_list[i], model->w[i]);
    }
}

void lr_model_destroy(lr_model *model)
{
    free(model);
}