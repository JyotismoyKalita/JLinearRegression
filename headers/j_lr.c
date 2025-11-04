#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "j_lr.h"

typedef struct lr_mdl
{
    float b;
    float l1;
    float l2;
    int count;
    unsigned char normalize;
    float *w;
    float *mean;
    float *std;
} lr_model;

float square(float x)
{
    return x * x;
}

static inline float soft_threshold(float z, float g)
{
    if (z > g)
        return z - g;
    if (z < -g)
        return z + g;
    return 0.0f;
}

numframe *lr_normalize_features(const numframe *x, lr_model *model)
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
        model->mean[j] = mean;
        float variance = (sumSq / rows) - (mean * mean);
        float std = sqrtf(variance);
        if (std == 0)
            std = 1;
        model->std[j] = std;
        for (int i = 0; i < rows; i++)
        {
            n_x->arr[i * cols + j] = (n_x->arr[i * cols + j] - mean) / std;
        }
    }
    return n_x;
}

numframe *lr_normalize_apply(const numframe *x, const lr_model *model)
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
        for (int i = 0; i < rows; i++)
        {
            n_x->arr[i * cols + j] = (n_x->arr[i * cols + j] - model->mean[j]) / model->std[j];
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

void gradient(float *dw, float *db, const numframe *x, const numframe *y, const float *w, const float b, const float l1, const float l2)
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
        dw[j] = t_dw[j] / (float)n + (l2 / n) * w[j];
    }
    *db = t_db / (float)n;
    free(t_dw);
}

void gradient_decent(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model)
{
    if (x == NULL || y == NULL)
        return;
    int cols = x->cols;
    float *w;
    w = (float *)malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; i++)
        w[i] = model->w[i];
    float b = model->b, cost;
    for (int i = 0; i < iter; i++)
    {
        float *dw, db = 0;
        dw = calloc(cols, sizeof(float));
        gradient(dw, &db, x, y, w, b, model->l1, model->l2);
        for (int j = 0; j < cols; j++)
        {
            w[j] = w[j] - alpha * dw[j];
        }
        b = b - alpha * db;
        cost = cost_function(x, y, w, b);
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
            printf("%f * %s + ", w_val, x->features[i]);
    }
    if (verbose > 1)
        printf("%.2f\n", b);
    free(w);
}

void lasso_coordinate_descent(numframe *x, const numframe *y, int iter, float lambda, lr_model *model)
{
    int rows = x->rows;
    int cols = x->cols;
    float *w = model->w;
    for (int it = 0; it < iter; it++)
    {
        for (int j = 0; j < cols; j++)
        {
            float rho = 0.0f;
            float denom = 0.0f;
            for (int i = 0; i < rows; i++)
            {
                float pred = model->b;
                for (int k = 0; k < cols; k++)
                    if (k != j)
                        pred += w[k] * x->arr[i * cols + k];

                rho += x->arr[i * cols + j] * (y->arr[i] - pred);
                denom += x->arr[i * cols + j] * x->arr[i * cols + j];
            }
            float z = rho / denom;
            float th = lambda / denom;
            if (z > th)
                w[j] = z - th;
            else if (z < -th)
                w[j] = z + th;
            else
                w[j] = 0.0f;
        }
        float sum = 0.0f;
        for (int i = 0; i < rows; i++)
        {
            float pred = model->b;
            for (int k = 0; k < cols; k++)
                pred += w[k] * x->arr[i * cols + k];
            sum += (y->arr[i] - pred);
        }
        model->b += sum / rows;
    }
}

void elastic_net_coordinate_descent(numframe *x, const numframe *y,
                                    float alpha, int iter,
                                    float l1, float l2, lr_model *model)
{
    int n = x->rows;
    int p = x->cols;

    float *w = model->w;
    float b = model->b;

    for (int it = 0; it < iter; it++)
    {
        // update bias: no L1/L2 penalty on intercept
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float pred = b;
            for (int j = 0; j < p; j++)
                pred += w[j] * x->arr[i * p + j];
            sum += (y->arr[i] - pred);
        }
        b += alpha * (sum / n);

        for (int j = 0; j < p; j++)
        {

            float rho = 0.0f;
            for (int i = 0; i < n; i++)
            {
                float pred = b;
                for (int k = 0; k < p; k++)
                    if (k != j)
                        pred += w[k] * x->arr[i * p + k];
                rho += x->arr[i * p + j] * (y->arr[i] - pred);
            }

            float z = rho / n;
            float denom = 1 + (l2 / n); // ridge part
            float w_new = soft_threshold(z, l1 / n) / denom;

            w[j] = w[j] + alpha * (w_new - w[j]); // smooth update
        }
    }

    model->b = b;
}

lr_model *lr_model_fit(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, const bool normalize, const float l1, const float l2)
{
    int cols = x->cols;
    int weight_size = sizeof(float) * cols;
    lr_model *model = (lr_model *)calloc(1, sizeof(lr_model));
    model->l1 = l1;
    model->l2 = l2;
    model->count = cols;
    model->normalize = normalize;
    model->w = (float *)calloc(cols, sizeof(float));
    model->mean = (float *)calloc(cols, sizeof(float));
    model->std = (float *)calloc(cols, sizeof(float));
    numframe *n_x = (model->normalize ? lr_normalize_features(x, model) : x);
    if (l1 > 0 && l2 > 0)
    {
        elastic_net_coordinate_descent(n_x, y, alpha, iter, model->l1, model->l2, model);
    }
    if (l1 > 0 && l2 == 0)
    {

        lasso_coordinate_descent(n_x, y, iter, l1, model);
    }
    else
    {
        gradient_decent(n_x, y, alpha, iter, minimum_cost, verbose, model);
    }
    if (model->normalize)
        nframe_destroy(n_x);
    return model;
}

void *lr_model_retrain(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, const float l1, const float l2, lr_model *model)
{
    model->l1 = l1;
    model->l2 = l2;
    gradient_decent(x, y, alpha, iter, minimum_cost, verbose, model);
}

numframe *lr_model_predict(numframe *x, const lr_model *model, const char *header)
{
    if (x == NULL)
        return NULL;
    numframe *n_x;
    if (model->normalize == 1)
        n_x = lr_normalize_apply(x, model);
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

float lr_r2_score(const numframe *predict, const numframe *y)
{
    int n = y->rows;
    if (n == 0)
        return 0;
    float ss_res = 0.0f;
    float ss_tot = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += y->arr[i];
    float mean = sum / n;
    for (int i = 0; i < n; i++)
    {
        ss_res += square(predict->arr[i] - y->arr[i]);
        ss_tot += square(y->arr[i] - mean);
    }
    if (ss_tot == 0)
        return 0;
    return 1.0f - (ss_res / ss_tot);
}

float lr_adjusted_r2_score(float r2, int n, int p)
{
    if (n <= p + 1)
    {
        return r2;
    }
    float adj_r2 = 1.0f - ((1.0f - r2) * ((float)(n - 1) / (float)(n - p - 1)));
    return adj_r2;
}

lr_model *lr_gridsearch(numframe *x, const numframe *y, const unsigned char normalize, const int iterations, const float *alphas, const int num_alphas, const float *l1, const int num_l1s, const float *l2, const int num_l2s)
{
    float best_adjusted_r2 = -9999;
    float best_alpha = 0;
    float best_l1 = 0;
    float best_l2 = 0;
    lr_model *best_model;
    for (int i = 0; i < num_alphas; i++)
    {
        for (int j = 0; j < num_l1s; j++)
        {
            for (int k = 0; k < num_l2s; k++)
            {
                float curr_alpha = alphas[i];
                float curr_l1 = l1[j];
                float curr_l2 = l2[k];
                printf("Trying: Alpha - %f, L1 = %f, L2 - %f\n", curr_alpha, curr_l1, curr_l2);
                lr_model *model = lr_model_fit(x, y, curr_alpha, iterations, 0, 0, normalize, curr_l1, curr_l2);
                numframe *predict = lr_model_predict(x, model, "Predictions");
                float r2 = lr_r2_score(predict, y);
                float adjusted_r2 = lr_adjusted_r2_score(r2, y->rows, x->cols);
                if (adjusted_r2 > best_adjusted_r2)
                {
                    best_model = model;
                    printf("This performed better\n");
                    best_adjusted_r2 = adjusted_r2;
                    best_alpha = curr_alpha;
                    best_l1 = curr_l1;
                    best_l2 = curr_l2;
                }
                else
                {
                    lr_model_destroy(model);
                }
                nframe_destroy(predict);
            }
        }
    }
    printf("Best Adjusted R2 = %f at Alpha - %f, L1 = %f, L2 - %f\n", best_adjusted_r2, best_alpha, best_l1, best_l2);
    return best_model;
}

float lr_kfold_score(numframe *x, numframe *y, int k,
                     float alpha, float l1, float l2,
                     unsigned char normalize, int iterations)
{
    int n = x->rows;
    int fold_size = n / k;

    float total_score = 0.0f;

    for (int fold = 0; fold < k; fold++)
    {
        int start = fold * fold_size;
        int end = (fold == k - 1) ? n : start + fold_size;

        numframe *x_val = nframe_slice_rows(x, start, end);
        numframe *y_val = nframe_slice_rows(y, start, end);

        numframe *x_train = nframe_exclude_rows(x, start, end);
        numframe *y_train = nframe_exclude_rows(y, start, end);

        lr_model *model = lr_model_fit(
            x_train, y_train,
            alpha, iterations, 0, 0,
            normalize, l1, l2);

        numframe *pred = lr_model_predict(x_val, model, "fold_pred");

        float r2 = lr_r2_score(pred, y_val);
        float adj = lr_adjusted_r2_score(r2, y_val->rows, x_val->cols);

        total_score += adj;

        lr_model_destroy(model);
        nframe_destroy(pred);
        nframe_destroy(x_val);
        nframe_destroy(y_val);
        nframe_destroy(x_train);
        nframe_destroy(y_train);
    }

    return total_score / k;
}

lr_model *lr_gridsearchcv(numframe *x, numframe *y,
                          unsigned char normalize, int iterations,
                          const float *alphas, int num_alphas,
                          const float *l1s, int num_l1s,
                          const float *l2s, int num_l2s,
                          int k)
{
    float best_score = -9999999;
    float best_alpha = 0, best_l1 = 0, best_l2 = 0;

    for (int i = 0; i < num_alphas; i++)
    {
        for (int j = 0; j < num_l1s; j++)
        {
            for (int k2 = 0; k2 < num_l2s; k2++)
            {
                float alpha = alphas[i];
                float l1 = l1s[j];
                float l2 = l2s[k2];

                float score = lr_kfold_score(x, y, k, alpha, l1, l2, normalize, iterations);

                printf("CV Alpha=%f L1=%f L2=%f Score=%f\n", alpha, l1, l2, score);

                if (score > best_score)
                {
                    best_score = score;
                    best_alpha = alpha;
                    best_l1 = l1;
                    best_l2 = l2;
                }
            }
        }
    }

    printf("\nBEST: AdjR2=%f | alpha=%f l1=%f l2=%f\n",
           best_score, best_alpha, best_l1, best_l2);

    return lr_model_fit(x, y, best_alpha, iterations, 0, 0, normalize, best_l1, best_l2);
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
    fwrite(&model->l1, sizeof(float), 1, mf);
    fwrite(&model->l2, sizeof(float), 1, mf);
    fwrite(&model->count, sizeof(int), 1, mf);
    fwrite(&model->normalize, sizeof(unsigned char), 1, mf);
    fwrite(model->w, sizeof(float), model->count, mf);
    fwrite(model->mean, sizeof(float), model->count, mf);
    fwrite(model->std, sizeof(float), model->count, mf);

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
    float b, l1, l2;
    int count;
    bool normalize;
    fread(&b, sizeof(float), 1, mf);
    fread(&l1, sizeof(float), 1, mf);
    fread(&l2, sizeof(float), 1, mf);
    fread(&count, sizeof(int), 1, mf);
    fread(&normalize, sizeof(bool), 1, mf);
    lr_model *model = (lr_model *)malloc(sizeof(lr_model));
    model->b = b;
    model->l1 = l1;
    model->l2 = l2;
    model->count = count;
    model->normalize = normalize;
    model->w = (float *)calloc(count, sizeof(float));
    fread(model->w, sizeof(float), count, mf);
    model->mean = (float *)calloc(count, sizeof(float));
    fread(model->mean, sizeof(float), count, mf);
    model->std = (float *)calloc(count, sizeof(float));
    fread(model->std, sizeof(float), count, mf);
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
    free(model->std);
    free(model->mean);
    free(model->w);
    free(model);
}