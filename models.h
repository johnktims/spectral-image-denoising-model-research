#ifndef MODELS_H
#define MODELS_H

#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif


typedef struct _options
{
    std::string method;
    int iterations;
    float var;
    bool add_noise;
} options;

template<class T> class Image
{
  public:
  IplImage* imgp;
  Image(IplImage* img=0) {imgp=img;}
  ~Image(){imgp=0;}
  void operator=(IplImage* img) {imgp=img;}
  inline T* operator[](const int rowIndx) {
    return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));}
};

typedef struct{
  unsigned char b,g,r;
} RgbPixel;

typedef struct{
  float b,g,r;
} RgbPixelFloat;

typedef Image<RgbPixel>       RgbImage;
typedef Image<RgbPixelFloat>  RgbImageFloat;
typedef Image<unsigned char>  BwImage;
typedef Image<float>          BwImageFloat;
typedef Image<double>         BwImageDouble;

double psnr(IplImage*, IplImage*);

// non-convex
void non_convex(IplImage*, IplImage*, int);

// nlm-naive
CvMat *cvGaussianKernel(int, float);
float cvGaussianWeightedDistance(CvMat*, IplImage *, IplImage *);
void cvGetSubImage(IplImage*, IplImage*, CvRect);
void nlm_naive(IplImage*, IplImage*);

// nlm-mean
template<class T> T si_sum(Image<T>, int, int, int);
void nlm_mean(IplImage*, IplImage*);

// noise
void addGaussianNoise(IplImage*, IplImage*, double, double);

#endif /* MODELS_H */

