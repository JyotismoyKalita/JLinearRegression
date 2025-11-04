#ifndef JNUMFRAME
#define JNUMFRAME

#define bool unsigned char

typedef struct nframe
{
    int rows;
    int cols;
    float *arr;
    char **features;
} numframe;

numframe *nframe_readcsv(const char *filename, const char seperator, const int line_length);

void nframe_show(const numframe *ndat);

void nframe_drop_inf_row(numframe *ndat);

void nframe_avg_inf_(numframe *ndat);

void nframe_shuffle(numframe *ndat, int seed);

void nframe_split_row(float ratio, numframe *original, numframe **ndat1, numframe **ndat2, bool shuffle, int seed);

numframe *nframe_split_col(const int start, const int end, const int y_index, const numframe *ndat);

numframe *nframe_join_y(numframe *ndat1, numframe *ndat2);

numframe *nframe_slice_rows(const numframe *ndat, int start, int end);

numframe *nframe_exclude_rows(const numframe *ndat, int start, int end);

void nframe_destroy(numframe *ndat);

#endif