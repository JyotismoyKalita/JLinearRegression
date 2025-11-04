#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "j_numframe.h"

void strip_char(char *str, int len)
{
    str[len - 1] = '\0';
    for (int i = 0; i < len; i++)
        str[i] = str[i + 1];
}

void *arr_rightshift(char *arr, int start, int *size)
{
    *size = *size + 1;
    for (int i = *size; i > start + 1; i--)
    {
        arr[i] = arr[i - 1];
    }
}

numframe *nframe_readcsv(const char *filename, const char seperator, const int line_length)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Error reading csv\n");
        return NULL;
    }
    char sep[2] = {seperator, '\0'};
    numframe *ndat = (numframe *)malloc(sizeof(numframe));
    char line[line_length];
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    int cols = 0;
    int rows = 0;
    rewind(fp);
    char rawdata[size * 2];
    memset(rawdata, 0, size * sizeof(char));
    while (fgets(line, line_length, fp) != NULL)
    {
        if (cols == 0)
        {
            line[strcspn(line, "\n")] = '\0';
            for (int i = 0; i < strlen(line); i++)
            {
                if (line[i] == seperator)
                    cols++;
            }
            cols++;
            ndat->features = (char **)malloc(sizeof(char *) * cols);
            char *token = strtok(line, sep);
            for (int i = 0; i < cols; i++)
            {
                int token_length = strlen(token);
                char temp[token_length];
                strcpy(temp, token);
                if (temp[0] == '\"' && temp[token_length - 1] == '\"')
                    strip_char(temp, token_length);
                if (strlen(temp) == 0)
                    sprintf(temp, "%d", i);
                ndat->features[i] = strdup(temp);
                token = strtok(0, sep);
            }
            ndat->features[cols - 1][strcspn(ndat->features[cols - 1], "\n")] = '\0';
            continue;
        }
        int n_idx = strcspn(line, "\n");
        line[n_idx] = seperator;
        line[n_idx + 1] = '\0';
        strcat(rawdata, line);
        rows++;
    }
    int raw_len = strlen(rawdata);
    for (int i = 0; i < raw_len - 1; i++)
    {
        if (rawdata[i] == ',' && rawdata[i + 1] == ',')
        {
            arr_rightshift(rawdata, i, &raw_len);
            rawdata[i + 1] = 'A';
        }
    }
    fclose(fp);
    ndat->cols = cols;
    ndat->rows = rows;
    ndat->arr = (float *)malloc(sizeof(float) * cols * rows);
    char *token = strtok(rawdata, sep);
    for (int i = 0; i < cols * rows; i++)
    {
        int token_length = strlen(token);
        char temp[token_length];
        strcpy(temp, token);
        if (temp[0] == '\"' && temp[token_length - 1] == '\"')
            strip_char(temp, token_length);
        if (strlen(temp) == 0)
        {
            ndat->arr[i] = FLT_MAX;
        }
        else
        {
            int res = sscanf(temp, "%f", &ndat->arr[i]);
            if (res == 0)
                ndat->arr[i] = FLT_MAX;
        }
        token = strtok(0, sep);
    }
    return ndat;
}

void nframe_show(const numframe *ndat)
{
    if (ndat == NULL)
        return;
    int cols = ndat->cols;
    int rows = ndat->rows;
    for (int i = 0; i < cols; i++)
    {
        printf("%-10s", ndat->features[i]);
    }
    printf("\n");
    for (int i = 0; i < cols * rows; i++)
    {
        if (ndat->arr[i] == FLT_MAX)
            printf("%-10s", "INF");
        else
            printf("%-10.2f", ndat->arr[i]);
        if ((i + 1) % cols == 0)
            printf("\n");
    }
}

void nframe_drop_inf_row(numframe *ndat)
{
    if (ndat == NULL)
        return;
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
    if (ndat == NULL)
        return;
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
    if (ndat == NULL)
        return;
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

void nframe_split_row(float ratio, numframe *original, numframe **ndat1, numframe **ndat2, bool shuffle, int seed)
{
    if (original == NULL)
        return;
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

numframe *nframe_split_col(const int start, const int end, const int y_index, const numframe *ndat)
{
    if (ndat == NULL)
        return NULL;
    int new_cols = end - start;
    if (!(y_index > start && y_index < end))
        new_cols++;
    int col = ndat->cols;
    int row = ndat->rows;
    numframe *newdat = (numframe *)malloc(sizeof(numframe));
    newdat->cols = new_cols;
    newdat->rows = row;
    newdat->features = (char **)malloc(sizeof(char *) * new_cols);
    newdat->arr = (float *)malloc(sizeof(float) * row * new_cols);
    int ni = 0;
    for (int i = start; i <= end; i++)
    {
        if (i == y_index)
            continue;
        newdat->features[ni++] = strdup(ndat->features[i]);
    }
    ni = 0;
    for (int i = start; i < col * row; i++)
    {
        if (i % col == y_index || i % col < start || i % col > end)
            continue;
        newdat->arr[ni++] = ndat->arr[i];
    }
    return newdat;
}

numframe *nframe_join_y(numframe *ndat1, numframe *ndat2)
{
    if (ndat1 == NULL || ndat2 == NULL)
        return NULL;
    if (ndat1->rows != ndat2->rows)
    {
        printf("Shape Do not Match to join\n");
        return NULL;
    }
    int cols1 = ndat1->cols;
    int cols2 = ndat2->cols;
    int new_cols = cols1 + cols2;
    int rows = ndat1->rows;
    int shape1 = cols1 * rows;
    int shape2 = cols2 * rows;
    numframe *newdat = (numframe *)malloc(sizeof(numframe));
    newdat->cols = new_cols;
    newdat->rows = rows;
    newdat->features = (char **)malloc(sizeof(char *) * new_cols);
    newdat->arr = (float *)malloc(sizeof(float) * rows * new_cols);
    for (int i = 0; i < cols1; i++)
    {
        newdat->features[i] = strdup(ndat1->features[i]);
    }
    for (int i = cols1; i < new_cols; i++)
    {
        newdat->features[i] = strdup(ndat2->features[i - cols1]);
    }
    int i1 = 0, i2 = 0;
    for (int i = 0; i < shape1 + shape2; i++)
    {
        if (i % new_cols < cols1)
            newdat->arr[i] = ndat1->arr[i1++];
        else
            newdat->arr[i] = ndat2->arr[i2++];
    }
    return newdat;
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