#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#include <iostream>

#include <stdio.h>  // printf
#include <ctype.h>  // isdigit
#include "models.h"

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

#define PI 3.14159265

#define S 21
#define P 7
#define H 10
#define SIGMA 6

using namespace std;
using namespace cv;

CvMat *cvGaussianKernel(int n, float sigma)
{
    if(n % 2 == 0)
    {
        return NULL;
    }

    sigma *= sigma;

    int mid = n / 2,
        x, y;
    float v;

    CvMat *ker = cvCreateMat(n, n, CV_32FC1);

    for(y = -mid; y <= mid; ++y)
    {
        for(x = -mid; x <= mid; ++x)
        {
            v  = exp(-(x * x + y * y)/(2 * sigma));
            v /= 2 * PI * sigma;
            cvmSet(ker, y + mid, x + mid, v);
        }
    }

    return ker;
}

float cvGaussianWeightedDistance(CvMat *kernel, IplImage *i, IplImage *j)
{
    float ret = 0;

    int y, x,
        t_mul,
        i_step = i->widthStep/sizeof(uchar), 
        k_step = kernel->cols;

    uchar *i_data = (uchar*)i->imageData;
    uchar *j_data = (uchar*)j->imageData;
    float *k_data = kernel->data.fl;

    for(y = 0; y < i->height; ++y)
    {
        for(x = 0; x < i->width; ++x)
        {
            t_mul  = i_data[y*i_step+x] - j_data[y*i_step+x];
            t_mul *= t_mul;
            ret   += k_data[y*k_step+x] * t_mul;
        }
    }

    return ret;
}

void cvPrintMatrix(CvMat *m)
{
    int y, x;

    for(y = 0; y < m->rows; ++y)
    {
        for(x = 0; x < m->cols; ++x)
        {
            printf("%f,", cvmGet(m, y, x));
        }
        printf("\n");
    }
}

void cvGetSubImage(IplImage* img, IplImage* subImg, CvRect roiRect)
{
    cvSetImageROI(img, roiRect);
    cvCopy(img, subImg, NULL);
    cvResetImageROI(img);
}

int main(int argc, char *argv[])
{
    // Create a Guassian Kernel that is the same size as the patch
    CvMat *kernel = cvGaussianKernel(P, SIGMA);
    cvPrintMatrix(cvGaussianKernel(5, 1));


    // Create patches and search windows
    IplImage *i_patch  = cvCreateImage(cvSize(P, P), IPL_DEPTH_8U, 1),
             *j_patch  = cvCreateImage(cvSize(P, P), IPL_DEPTH_8U, 1);


    // Initialize windows
    cvNamedWindow(WIN_ORIGINAL, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(WIN_ORIGINAL, 0, 0);

    cvNamedWindow(WIN_MODIFIED, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(WIN_MODIFIED, 200, 200);


    // Load and convert image to grayscale
    IplImage *original      = cvLoadImage(argv[1], -1),
             *gray_original = NULL;

    if(original->nChannels > 1)
    {
        gray_original = cvCreateImage(cvGetSize(original), original->depth, 1);
        cvCvtColor(original, gray_original, CV_BGR2GRAY);
    }
    else
    {
        gray_original = original;
    }

    printf("Original image: w:%dx h:%d\n", gray_original->width, gray_original->height);

    // Prepare for convolution
    IplImage *padded = cvCreateImage(cvSize(gray_original->width+S+P,
                                     gray_original->height+S+P),
                                     gray_original->depth,
                                     gray_original->nChannels),
             *output = cvCreateImage(cvGetSize(gray_original), IPL_DEPTH_8U, gray_original->nChannels);

    printf("Padded image: w:%dx h:%d\n", padded->width, padded->height);

    CvMat *weight = cvCreateMat(S, S, CV_32FC1);

    // Pad borders for convolution
    CvPoint offset = cvPoint((S+P-1)/2,(S+P-1)/2);
    cvCopyMakeBorder(gray_original, padded, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));


    BwImage u(output);
    BwImage f(padded);

    int x0 = (P+S)/2, xn = padded->width  - (P+S)/2,
        y0 = (P+S)/2, yn = padded->height - (P+S)/2,
        sx, sy;

    float f_x=0.0,
          c_x=0.0,
          w_x=0.0,
          tmp=0.0;

    for(y0 = (P+S)/2; y0 < yn; ++y0)
    {
        for(x0 = (P+S)/2; x0 < xn; ++x0)
        {
            f_x = 0.0;
            c_x = 0.0;

            // Save patch around current pixel
            cvGetSubImage(padded, i_patch, cvRect(x0-P/2, y0-P/2, P, P));
            
            // Determine weighted distance for each patch as well as normalized constant
            for(sy = y0-S/2; sy < y0+S/2+1; ++sy)
            {
                for(sx = x0-S/2; sx < x0+S/2+1; ++sx)
                {
                    // Save patch around current iteration in the search window
                    cvGetSubImage(padded, j_patch, cvRect(sx-P/2, sy-P/2, P, P));

                    cvmSet(weight, sy+S/2-y0, sx+S/2-x0, exp(-cvGaussianWeightedDistance(kernel, i_patch, j_patch)/(H*H)));
                    c_x += cvmGet(weight, sy+S/2-y0, sx+S/2-x0);
                }
            }

            for(sy = y0-S/2; sy < y0+S/2+1; ++sy)
            {
                for(sx = x0-S/2; sx < x0+S/2+1; ++sx)
                {
                    tmp = cvmGet(weight, sy+S/2-y0, sx+S/2-x0) * f[sy][sx];
                    f_x += tmp;
                }
            }
            u[y0-(P+S)/2][x0-(P+S)/2] = (int)f_x/c_x;
        }
    }

    /*
    cvShowImage(WIN_ORIGINAL, gray_original);
    cvShowImage(WIN_MODIFIED, output);

    cvWaitKey(0);
    */

    cvSaveImage("out.jpg", output);

    return 0;
}

