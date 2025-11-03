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

void nframe_show(numframe *);

void nframe_drop_inf_row(numframe *);

void nframe_avg_inf_(numframe *);

void nframe_shuffle(numframe *, int seed);

void nframe_split(float ratio, numframe *original, numframe **ndat1, numframe **ndat2, bool shuffle, int seed);

void nframe_destroy(numframe *);

#endif