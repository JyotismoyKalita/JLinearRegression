#include <stdio.h>
#include "../headers/j_lr.h"

int main()
{
    numframe *ndat = nframe_readcsv("sample.csv", ',', 128);
    numframe *x_train, *y_train, *x_test, *y_test;
    lr_train_test_split(&x_train, &y_train, &x_test, &y_test, ndat, 0.8, 1, 42, 0, 1, 2);

    lr_model *model = lr_model_fit(x_train, y_train, 0.001, 10000, 0.0001, 0);
    numframe *predict = lr_model_predict(x_test, model, "Predicted");
    numframe *comparison = nframe_join_y(predict, y_test);
    nframe_show(comparison);
    float mse = lr_calculate_mse(predict, y_test);
    printf("Test MSE: %f\n", mse);
    return 0;
}