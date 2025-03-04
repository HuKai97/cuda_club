#include <stdlib.h>


bool check(float *h_result, float *h_output, int n)
{
    for (int i = 0; i < n; i++)
    {
        if(abs(h_result[i] - h_output[i]) > 1e-4)
        {
            return false;
        }
    }
    return true;
}