# JLinearRegression - j_lr.h & j_numframe.h

Header files built in C to perform Linear Regression  
_- by Jyotismoy Kalita_  

## Usage

### Cloning

```sh
git clone "https://github.com/JyotismoyKalita/JLinearRegression.git"
cd JLinearRegression
```

The header files are located in the `headers` folder.  
The example fils are located in the `example` folder.

### Including in C File

To use the **Numframe** header file only, write in your c file:

```c
#include "path/to/j_numframe.h"
```

When using the **Linear Regression** header file, we dont need to seperately include the **Numframe** header file.

```c
#include "path/to/j_lr.h"
```

### Executing

If you are only using the **Numframe** header file:

```sh
gcc example.c path/to/j_numframe.c -o example
./example
```

If you are also using the **Linear Regression** header file:

```sh
gcc example.c path/to/j_lr.c path/to/j_numframe.c -lm -o example
./example
```

## j_numframe.h

The `j_numframe.h` is used to import `.csv` files into the memory and to perform operations on them. This is later required by the linear regression algorithms.  

It uses a user built struct:

```c
typedef struct nframe
{
    int rows;
    int cols;
    float *arr;
    char **features;
} numframe;
```

`int rows` and `int cols` store the number of rows and columns of the .csv file respectively. It excludes the row holdig the column names in the count.

`float arr` stores the actual data as a 1-dimensional array.  
`char **features` stores an array of the column names in the csv.  

### Functions-NF

---

```c
numframe *nframe_readcsv(const char *filename, const char seperator, const int line_length);
```

Opens a `.csv` file and loads it into the memory using the `numframe` struct. The first row of the .csv file is always assumed to be the row holding column names.  
It takes name of the file `const char *filename`, seperator used in the .csv `const char seperator` and the maximum size of line buffer(to accomodate long lines/rows the csv) `cont int line_length` as argument.  

---

```c
void nframe_show(const numframe *ndat);
```

Prints the entire table/dataframe/numframe.

---

```c
void nframe_drop_inf_row(numframe *ndat);
```

Drops rows in the numframe, containing `INF` values. These `INF` values are generated at the index where there were invalid values in the csv.

---

```c
void nframe_avg_inf_(numframe *ndat);
```

Averages the values at the indexes containing `INF` values.

---

```c
void nframe_shuffle(numframe *ndat, int seed);
```

Shuffles the position of rows randomly based on the seed passed as argument.

---

```c
void nframe_split_row(float ratio, numframe *original, numframe **ndat1, numframe **ndat2, bool shuffle, int seed);
```

Splits the numframe vertically in the `ratio` passed as argument. Generates two new numframes, the pointers to which is passed on to the empty numframe pointers passed in the argument `numframe **ndat1` and `numframe **ndat2`. Shuffling can be enabled by passing shuffle argument as 1 and a seed value.

---

```c
numframe *nframe_split_col(const int start, const int end, const int y_index, const numframe *ndat);
```

Generates a new numframe with the columns from `start` to `end` in the original numframe `ndat`, excluding column in between if it is the output column i.e. its index is the `y_index`. It returns the address to the new numframe.

---

```c
numframe *nframe_join_y(numframe *ndat1, numframe *ndat2);
```

Combines two numframes along the y-axis i.e. combines all of their columns together into one numframe. Their rows must match to perform this operation.

---

```c
numframe *nframe_slice_rows(const numframe *ndat, int start, int end);
```

Returns a new numframe from `start` to `end` of `ndat`

---

```c
numframe *nframe_exclude_rows(const numframe *ndat, int start, int end);
```

Returns a new numframe excluding the rows from `start` to `end` of `ndat`

---

```c
void nframe_destroy(numframe *ndat);
```

Every generated numframe must be freed with this function to avoid memory leak. `DO NOT` manually free the numframes.

## j_lr.h

The `j_lr.h` contains the functions to perform linear regressions on the numframes. It strictly uses the _gradient descent_ algorithm for performing linear regression.

It also allows to `save` the generated model. `load` a pretrained model and `retrain` a pretrained model.

The model is a struct file:

```c
typedef struct lr_mdl
{
    float b;
    int count;
    unsigned char normalize;
    float w[];
} lr_model;
```

`float b` is the bias. `int count` is the count of weights. `float w[]` is a flexible array storing the value of weights. `unsigned char normalize` keeps track if model needs to normalize the data. However this struct is not directly accessible. And can only be generated, saved, loaded, trained, retrained using the functions to avoid corruption of data.

### Functions-LR  

---

```c
void lr_train_test_split(numframe **x_train, numframe **y_train, numframe **x_test, numframe **y_test, numframe *original, const float ratio, const bool shuffle, const int seed, const int start, const int end, const int y_index);
```

Splits the original numframe into x and y and their train and test splits. Four new numframes are generated whose address are assigned to the pointers to emptpy numframes passed as arguments in the functions: `x_train`, `x_test`, `y_train` and `y_test`. The split is done in `float ratio` and shuffling can be enabled by putting `shuffle` argument as 1 and using a seed. Specific columns can be selected for the final numframes using the `start` and `end` index. `y_index` indicates the column containig output values. This column will be ignored in the `x_train` and `x_test` numframe.

---

```c
lr_model *lr_model_fit(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, const bool normalize);
```

Fits a model to the training data. `x` is the input numframe. `y` is the numframe containing output values for training. `alpha` is the learning rate. `iter` is the maximum iterations to perform. `minimum_cost` is the minimum cost reaching which, the iterations stops. Setting `verbose` to

- 0: Displays nothing
- 1: Displays Train MSE
- 2: Displays the Final Function
- 3: Displays cost at each iteration

`normalize` parameter specifies whether the data needs to be normalizd. This returns an `lr_model` pointer pointing to the generated model.

---

```c
void *lr_model_retrain(const numframe *x, const numframe *y, const float alpha, const int iter, const float minimum_cost, const bool verbose, lr_model *model);
```

Same as `lr_model_fit`, but the difference being, this time it continues to train from the values of the pretrained `model` passed as argument.

---

```c
numframe *lr_model_predict(const numframe *x, const lr_model *model, const char *header);
```

Generates a new numframe with predicted value using a pretrained `model` on numframe `x`. The name of the column of the numframe is passed in the `header` argument.

---

```c
float lr_calculate_mse(const numframe *predict, const numframe *y);
```

Returns the Mean Sqaured Error value calculated using a numframe containig the predicted values `predict` and a numframe containing the actual values `y`.

```c
float lr_r2_score(const numframe *predict, const numframe *y);
```

Returns R2 score calculated using a numframe containig the predicted values `predict` and a numframe containing the actual values `y`.

```c
float lr_adjusted_r2_score(float r2, int n, int p);
```

Returns Adjusted R2 score calculated using the `r2` value and no of sampes `n` and no of features `p`.

---

```c
lr_model *lr_gridsearch(numframe *x, const numframe *y, const unsigned char normalize, const int iterations, const float *alphas, const int num_alphas, const float *l1, const int num_l1s, const float *l2, const int num_l2s);
```

Searches for the best parameters with respect to R2 score generated using them. Pass array containing values of the parameter e.g. `*alphas` and the no. of elements in them `num_alphas`. This returns the model trained with those parameters which gave the best R2.

---

```c
float lr_kfold_score(numframe *x, numframe *y, int k,
                     float alpha, float l1, float l2,
                     unsigned char normalize, int iterations);
```

Returns K-Fold score for input `x` and `y` with `k` folds and other required parameters.

---

```c
lr_model *lr_gridsearchcv(numframe *x, numframe *y,
                          unsigned char normalize, int iterations,
                          const float *alphas, int num_alphas,
                          const float *l1s, int num_l1s,
                          const float *l2s, int num_l2s,
                          int k);
```

Searches for best parameters with Cross-Validation to avoid Overfitting on the training data.

---

```c
void lr_model_save(const char *filename, const lr_model *model);
```

Saves the `model` struct into a binary file with name `filename`.

---

```c
lr_model *lr_model_load(const char *filename);
```

Loads a saved model binary file into the memory through the `lr_model` struct.

---

```c
void *lr_model_show_coeff(const lr_model *model, char **feature_list);
```

Displays the coefficients in `model` for each feature in `feature_list`.

---

```c
void lr_model_destroy(lr_model *model);
```

Every generated model must be freed using this function to avoid memory leakage. `DO NOT` manually free the model structs.
