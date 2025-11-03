#ifndef J_LR
#define J_LR
#include "j_numframe.h"

typedef struct lr_mdl lr_model;

float square(float x);

void lr_train_test_split(numframe **x_train, numframe **y_train, numframe **x_test, numframe **y_test, numframe *original, const float ratio, const bool shuffle, const int seed, const int start, const int end, const int y_index);

float cost_function(const numframe *x, const numframe *y, const float *w, const float b);

void gradient(float *dw, float *db, const numframe *x, const numframe *y, const float *w, const float b);

void gradient_decent(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model);

lr_model *lr_model_fit(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose);

void *lr_model_retrain(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model);

numframe *lr_model_predict(const numframe *x, const lr_model *model, const char *header);

float lr_calculate_mse(const numframe *predict, const numframe *y);

void lr_model_save(const char *filename, const lr_model *model);

lr_model *lr_model_load(const char *filename);

void lr_model_destroy(lr_model *model);

#endif