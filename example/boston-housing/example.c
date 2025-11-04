#include <stdio.h>
#include "../../headers/j_lr.h"
int main()
{
    numframe *raw = nframe_readcsv("Boston.csv", ',', 256);
    numframe *x_train, *x_test, *y_train, *y_test;
    lr_train_test_split(&x_train, &y_train, &x_test, &y_test, raw, 0.8, 1, 42, 1, 13, 14);
    nframe_destroy(raw);

    float alphas[1] = {0.001};
    float l1s[4] = {0, 0.01, 0.1, 1};
    float l2s[4] = {0, 0.01, 0.1, 1};

    lr_model *model = lr_gridsearchcv(x_train, y_train, 1, 10000, alphas, 1, l1s, 4, l2s, 4, 4);
    lr_model_save("model.bin", model);
    lr_model_destroy(model);

    lr_model *model2 = lr_model_load("model.bin");
    numframe *predict = lr_model_predict(x_test, model2, "Prediction");
    float mse = lr_calculate_mse(predict, y_test);
    float r2 = lr_r2_score(predict, y_test);
    float adjusted_r2 = lr_adjusted_r2_score(r2, y_test->rows, x_test->cols);
    nframe_destroy(predict);
    printf("TEST MSE: %f\nTEST R2: %f\nTEST Adjusted R2: %f\n", mse, r2, adjusted_r2);
    lr_model_show_coeff(model2, x_test->features);

    nframe_destroy(x_train);
    nframe_destroy(x_test);
    nframe_destroy(y_train);
    nframe_destroy(y_test);
    lr_model_destroy(model2);
    return 0;
}