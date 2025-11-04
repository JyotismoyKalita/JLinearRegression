#ifndef J_LR
#define J_LR
#include "j_numframe.h"

typedef struct lr_mdl lr_model;

void lr_train_test_split(numframe **x_train, numframe **y_train, numframe **x_test, numframe **y_test, numframe *original, const float ratio, const bool shuffle, const int seed, const int start, const int end, const int y_index);

lr_model *lr_model_fit(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, const bool normalize, const float l1, const float l2);

void *lr_model_retrain(numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, const float l1, const float l2, lr_model *model);

numframe *lr_model_predict(numframe *x, const lr_model *model, const char *header);

float lr_calculate_mse(const numframe *predict, const numframe *y);

float lr_r2_score(const numframe *predict, const numframe *y);

float lr_adjusted_r2_score(float r2, int n, int p);

lr_model *lr_gridsearch(numframe *x, const numframe *y, const unsigned char normalize, const int iterations, const float *alphas, const int num_alphas, const float *l1, const int num_l1s, const float *l2, const int num_l2s);

float lr_kfold_score(numframe *x, numframe *y, int k,
                     float alpha, float l1, float l2,
                     unsigned char normalize, int iterations);

lr_model *lr_gridsearchcv(numframe *x, numframe *y,
                          unsigned char normalize, int iterations,
                          const float *alphas, int num_alphas,
                          const float *l1s, int num_l1s,
                          const float *l2s, int num_l2s,
                          int k);

void lr_model_save(const char *filename, const lr_model *model);

lr_model *lr_model_load(const char *filename);

void *lr_model_show_coeff(const lr_model *model, char **feature_list);

void lr_model_destroy(lr_model *model);

#endif