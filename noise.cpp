#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#include <stdio.h>  // printf
#include <ctype.h>  // isdigit
#include <stdlib.h>
#include "models.h"

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

#define unit_random() (1.0*rand()/RAND_MAX)
#define TWO_PI 6.28318530717958647688

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

using namespace std;


IplImage* loadGrayscale(const string path)
{
    IplImage *src = cvLoadImage(path.c_str(), -1);

    if(!src)
    {
        return NULL;
    }

    IplImage *dst = cvCreateImage(cvSize(src->width, src->height),IPL_DEPTH_8U, 1);

    cvConvertImage(src, dst, 0);

    cvReleaseImage(&src);

    return dst;
}

void addNoise(IplImage *src, IplImage *dst)
{

}

void addGaussianNoise(IplImage *image_in, IplImage *image_out, double mean, double var)
{
    double stddev = sqrt(var);
    CvRNG rng_state = cvRNG(-1);

    cvRandArr(&rng_state, image_out, CV_RAND_NORMAL,
            cvRealScalar(mean*255), cvRealScalar(stddev*255));
    cvAdd(image_in, image_out, image_out);
}

int main(int argc, char *argv[])
{
    IplImage *src = loadGrayscale(argv[1]);
    IplImage *dst = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

    cvNamedWindow(WIN_ORIGINAL, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(WIN_ORIGINAL, 0, 0);

    cvNamedWindow(WIN_MODIFIED, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(WIN_MODIFIED, 0, 700);

    addGaussianNoise(src, dst, 0, 0.01);

    cvShowImage(WIN_ORIGINAL, src);
    cvShowImage(WIN_MODIFIED, dst);

    cvWaitKey(0);
    return 0;
}

