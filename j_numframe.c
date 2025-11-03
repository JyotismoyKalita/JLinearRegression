#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "j_numframe.h"

numframe *nframe_readcsv(const char *filename, const char seperator, const int line_length)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Error reading csv\n");
        return NULL;
    }
    numframe *ndat = (numframe *)malloc(sizeof(numframe));
    char line[line_length];
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    int cols = 0;
    int rows = 0;
    rewind(fp);
    char rawdata[size];
    memset(rawdata, 0, size);
    while (fgets(line, line_length, fp) != NULL)
    {
        if (cols == 0)
        {
            for (int i = 0; i < strlen(line); i++)
            {
                if (line[i] == seperator)
                    cols++;
            }
            cols++;
            ndat->features = (char **)malloc(sizeof(char *) * cols);
            char *token = strtok(line, &seperator);
            for (int i = 0; i < cols; i++)
            {
                ndat->features[i] = strdup(token);
                token = strtok(0, &seperator);
            }
            ndat->features[cols - 1][strcspn(ndat->features[cols - 1], "\n")] = '\0';
            continue;
        }
        line[strcspn(line, "\n")] = seperator;
        int line_first_comma = strcspn(line, ",");
        if (line[line_first_comma + 1] == ',')
        {
            line[line_first_comma + 1] = 'A';
            line[line_first_comma + 2] = ',';
            line[line_first_comma + 3] = '\0';
        }
        strcat(rawdata, line);
        rows++;
    }
    fclose(fp);
    ndat->cols = cols;
    ndat->rows = rows;
    ndat->arr = (float *)malloc(sizeof(float) * cols * rows);
    char *token = strtok(rawdata, &seperator);
    for (int i = 0; i < cols * rows; i++)
    {
        int res = sscanf(token, "%f", &ndat->arr[i]);
        if (res == 0)
            ndat->arr[i] = FLT_MAX;
        token = strtok(0, &seperator);
    }
    return ndat;
}

void nframe_show(numframe *ndat)
{
    int cols = ndat->cols;
    int rows = ndat->rows;
    for (int i = 0; i < cols; i++)
    {
        printf("%s\t", ndat->features[i]);
    }
    printf("\n");
    for (int i = 0; i < cols * rows; i++)
    {
        if (ndat->arr[i] == FLT_MAX)
            printf("Inf\t");
        else
            printf("%.2f\t", ndat->arr[i]);
        if ((i + 1) % cols == 0)
            printf("\n");
    }
}

void nframe_drop_inf_row(numframe *ndat)
{
    int cols = ndat->cols;
    int rows = ndat->rows;
    for (int i = 0; i < cols * rows; i++)
    {
        if (ndat->arr[i] == FLT_MAX)
        {
            int row_no = i / cols;
            for (int j = row_no * cols; j < cols * (rows - 1); j++)
            {
                ndat->arr[j] = ndat->arr[j + cols];
            }
            ndat->rows--;
            rows--;
        }
    }
}

void nframe_avg_inf_(numframe *ndat)
{
    int cols = ndat->cols;
    int rows = ndat->rows;
    float res = 0;
    int curr_col = -1;
    for (int i = 0; i < cols * rows; i++)
    {
        if (ndat->arr[i] == FLT_MAX)
        {
            if (curr_col != i % cols)
            {
                int col_no = i % cols;
                float sum = 0;
                int count = 0;
                for (int j = 0; j < rows; j++)
                {
                    if (ndat->arr[j * cols + col_no] != FLT_MAX)
                    {
                        sum += ndat->arr[j * cols + col_no];
                        count++;
                    }
                }
                res = sum / (float)count;
                ndat->arr[i] = res;
            }
            else
            {
                ndat->arr[i] = res;
            }
        }
    }
}

void nframe_shuffle(numframe *ndat, int seed)
{
    int rows = ndat->rows;
    int cols = ndat->cols;
    srand(seed);
    for (int i = 0; i < rows; i++)
    {
        int from = rand() % rows;
        int to = rand() % rows;
        float temp;
        for (int i = 0; i < cols; i++)
        {
            temp = ndat->arr[to * cols + i];
            ndat->arr[to * cols + i] = ndat->arr[from * cols + i];
            ndat->arr[from * cols + i] = temp;
        }
    }
}

void nframe_split(float ratio, numframe *original, numframe **ndat1, numframe **ndat2, bool shuffle, int seed)
{
    if (shuffle == 1)
        nframe_shuffle(original, seed);
    int cols = original->cols;
    int size1 = ratio * original->rows;
    int size2 = original->rows - size1;

    (*ndat1) = (numframe *)malloc(sizeof(numframe));
    (*ndat2) = (numframe *)malloc(sizeof(numframe));

    (*ndat1)->cols = cols;
    (*ndat1)->rows = size1;
    (*ndat1)->features = (char **)malloc(sizeof(char *) * cols);
    for (int i = 0; i < cols; i++)
        (*ndat1)->features[i] = strdup(original->features[i]);
    int shape1 = (*ndat1)->rows * cols;
    (*ndat1)->arr = (float *)malloc(sizeof(float) * shape1);
    for (int i = 0; i < shape1; i++)
        (*ndat1)->arr[i] = original->arr[i];

    (*ndat2)->cols = cols;
    (*ndat2)->rows = size2;
    (*ndat2)->features = (char **)malloc(sizeof(char *) * cols);
    for (int i = 0; i < cols; i++)
        (*ndat2)->features[i] = strdup(original->features[i]);
    int shape2 = (*ndat2)->rows * cols;
    (*ndat2)->arr = (float *)malloc(sizeof(float) * shape2);
    for (int i = shape1; i < original->rows * cols; i++)
        (*ndat2)->arr[i - shape1] = original->arr[i];
}

void nframe_destroy(numframe *ndat)
{
    if (ndat == NULL)
        return;
    for (int i = 0; i < ndat->cols; i++)
    {
        free(ndat->features[i]);
    }
    free(ndat->features);
    free(ndat->arr);
    free(ndat);
}