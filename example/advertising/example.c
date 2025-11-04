#include <stdio.h>
#include "../../headers/j_lr.h"

int main()
{
    numframe *raw = nframe_readcsv("advertising.csv", ',', 32);
    numframe *x_train, *y_train, *x_test, *y_test;
    lr_train_test_split(&x_train, &y_train, &x_test, &y_test, raw, 0.7, 1, 42, 0, 2, 3);
    nframe_destroy(raw);

    float alphas[1] = {0.001};
    float l1s[4] = {0, 0.01, 0.1, 1};
    float l2s[4] = {0, 0.01, 0.1, 1};

    lr_model *model = lr_gridsearchcv(x_train, y_train, 1, 10000, alphas, 1, l1s, 4, l2s, 4, 4);
    numframe *predict = lr_model_predict(x_test, model, "Prediction");
    numframe *combined = nframe_join_y(predict, y_test);
    float mse = lr_calculate_mse(predict, y_test);
    float r2 = lr_r2_score(predict, y_test);
    float adjusted_r2 = lr_adjusted_r2_score(r2, y_test->rows, x_test->cols);
    nframe_show(combined);
    printf("Test MSE: %f\nTest R2: %f\nTest Adjusted R2: %f\n", mse, r2, adjusted_r2);
    lr_model_show_coeff(model, x_test->features);

    nframe_destroy(combined);
    nframe_destroy(x_train);
    nframe_destroy(x_test);
    nframe_destroy(y_test);
    nframe_destroy(y_train);
    nframe_destroy(predict);
    lr_model_destroy(model);
    return 0;
}