#include <stdio.h>
#include "../../headers/j_lr.h"
int main()
{
    numframe *raw = nframe_readcsv("Boston.csv", ',', 256);
    numframe *x_train, *x_test, *y_train, *y_test;
    lr_train_test_split(&x_train, &y_train, &x_test, &y_test, raw, 0.8, 1, 42, 1, 13, 14);
    nframe_destroy(raw);

    lr_model *model = lr_model_fit(x_train, y_train, 0.001, 50000, 0, 1, 1);
    lr_model_save("model.bin", model);
    lr_model_destroy(model);

    lr_model *model2 = lr_model_load("model.bin");
    numframe *predict = lr_model_predict(x_test, model2, "Prediction");
    float mse = lr_calculate_mse(predict, y_test);
    nframe_destroy(predict);
    printf("TEST MSE: %f\n", mse);
    lr_model_show_coeff(model2, x_test->features);

    nframe_destroy(x_train);
    nframe_destroy(x_test);
    nframe_destroy(y_train);
    nframe_destroy(y_test);
    lr_model_destroy(model2);
    return 0;
}