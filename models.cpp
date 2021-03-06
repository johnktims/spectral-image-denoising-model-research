#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#define PI 3.14159265

// NLM-Naive
#define NLM_NAIVE_S 21
#define NLM_NAIVE_P 7
#define NLM_NAIVE_H 9
#define NLM_NAIVE_SIGMA 6

// NLM-Mean
#define NLM_MEAN_P1 3
#define NLM_MEAN_P2 5
#define NLM_MEAN_P3 7
#define NLM_MEAN_S  21
#define NLM_MEAN_A1 20
#define NLM_MEAN_A2 20
#define NLM_MEAN_A3 20
#define NLM_MEAN_SIGMA 6

#include <stdio.h>  // printf
#include <math.h>
#include "models.h"

/****************************************************************************
 * @brief Calculate the PSNR of the two given images
 * @return The PSNR value
 ****************************************************************************/
double psnr(IplImage *original, IplImage *output)
{
    BwImage f(original);
    BwImage u(output);

    int x,y,
        cols = original->width,
        rows = original->height;

    double mse = 0.0;
    for(y = 0; y < rows; ++y)
    {
        for(x = 0; x < cols; ++x)
        {
            mse += pow(f[y][x] - u[y][x], 2);
        }
    }
    mse /= cols * rows * 1.0;

    return 10*log10(255.0 * 255.0 / mse);
}


/****************************************************************************
 * @brief Perform N iterations of the non-convex model
 ****************************************************************************/
void non_convex(IplImage* f_t, IplImage* un_t, int N)
{
    IplImage* u_t = cvCreateImage(cvGetSize(f_t), f_t->depth, f_t->nChannels);
    
    BwImage u(u_t);
    BwImage f(f_t);
    BwImage un(un_t);
    
    int x,
        y,
        k;

    double ux,
           uy,
           star,
           a[4],
           d[4],

           epsilon = 0.05,
           lambda  = 0.6,
           t       = 0.1,
           p       = 0.1,

           exponent = ((2-p)/2.0);

    epsilon = pow(epsilon, 2);

    // TODO: Handle Neumann Boundary Conditions
    for(k = 0; k < N; ++k)
    {
        for(y = 1; y < f_t->width-1; ++y)
        {
            for(x = 1; x < f_t->height-1; ++x)
            {
                // a_{i+.5}
                ux = un[x+1][y]   - un[x][y];
                uy = un[x+1][y+1] + un[x][y+1] -
                     un[x+1][y-1] - un[x][y-1];
                ux = pow(ux, 2);
                uy = pow(uy / 4.0, 2);
                d[0] = pow(epsilon + ux + uy, exponent);

                // a_{i-.5}
                ux = un[x-1][y]   - un[x][y];
                uy = un[x-1][y+1] + un[x][y+1] -
                     un[x-1][y-1] - un[x][y-1];
                ux = pow(ux, 2);
                uy = pow(uy / 4.0, 2);
                d[1] = pow(epsilon + ux + uy, exponent);

                // a_{j+.5}
                uy = un[x][y+1]   - un[x][y];
                ux = un[x+1][y+1] + un[x+1][y] -
                     un[x-1][y+1] - un[x-1][y];
                uy = pow(uy, 2);
                ux = pow(ux / 4.0, 2);
                d[2] = pow(epsilon + ux + uy, exponent);

                // a_{j-.5}
                uy = un[x][y-1]   - un[x][y];
                ux = un[x+1][y-1] + un[x+1][y] -
                     un[x-1][y-1] - un[x-1][y];
                uy = pow(uy, 2);
                ux = pow(ux / 4.0, 2);
                d[3] = pow(epsilon + ux + uy, exponent);

                a[0] = 2.0*d[1]/(d[0] + d[1]);
                a[1] = 2.0*d[0]/(d[0] + d[1]);
                a[2] = 2.0*d[3]/(d[2] + d[3]);
                a[3] = 2.0*d[2]/(d[2] + d[3]);

                star = (
                        a[0]*un[x+1][y]  +
                        a[1]*un[x-1][y]  +
                        a[2]*un[x][y+1]  +
                        a[3]*un[x][y-1]
                       )
                       -
                       ((a[0] + a[1] + a[2] + a[3]) *
                        un[x][y]
                       ) +
                       lambda *
                       (f[x][y] - un[x][y]);
                u[x][y] = un[x][y] + t * star;
            }
        }
        cvCopy(u_t, un_t, NULL);
    }
}


/****************************************************************************
 * @brief Create a gaussian kernel matrix of size `n` and variance `sigma`
 * @return An NxN matrix
 ****************************************************************************/
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
            //norm += v;
        }
    }

    // It is unnecessary to normalize the kernel
    /*
    for(y = -mid; y <= mid; ++y)
    {
        for(x = -mid; x <= mid; ++x)
        {
            cvmSet(ker, y + mid, x + mid, cvmGet(ker, y + mid, x + mid)/norm);
        }
    }
    */

    return ker;
}


/****************************************************************************
 * @brief Calculate the Gaussian Weighted Distance between `i` and `j` using
 *        the previously calculated Gaussian kernel.
 * @return The Gaussian Weighted Distance
 ****************************************************************************/
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


/****************************************************************************
 * @brief  Copies a subimage into an array
 ****************************************************************************/
void cvGetSubImage(IplImage* img, IplImage* subImg, CvRect roiRect)
{
    cvSetImageROI(img, roiRect);
    cvCopy(img, subImg, NULL);
    cvResetImageROI(img);
}


/****************************************************************************
 * @brief Perform the Buades version of the Non-Local Mean Method
 ****************************************************************************/
void nlm_naive(IplImage *original, IplImage *output)
{
    // Create a Guassian Kernel that is the same size as the patches
    static CvMat *kernel = cvGaussianKernel(NLM_NAIVE_P, NLM_NAIVE_SIGMA);

    // Create patches
    static IplImage *i_patch = cvCreateImage(cvSize(NLM_NAIVE_P, NLM_NAIVE_P), IPL_DEPTH_8U, 1),
                    *j_patch = cvCreateImage(cvSize(NLM_NAIVE_P, NLM_NAIVE_P), IPL_DEPTH_8U, 1);

    // Create padded matrix
    static IplImage *padded = cvCreateImage(cvSize(original->width+NLM_NAIVE_S+NLM_NAIVE_P,
                                     original->height+NLM_NAIVE_S+NLM_NAIVE_P),
                                     original->depth,
                                     original->nChannels);

    // Create weight matrix
    static CvMat *weight = cvCreateMat(NLM_NAIVE_S, NLM_NAIVE_S, CV_32FC1);

    // Insert original image into padded matrix
    CvPoint offset = cvPoint((NLM_NAIVE_S+NLM_NAIVE_P-1)/2,(NLM_NAIVE_S+NLM_NAIVE_P-1)/2);
    cvCopyMakeBorder(original, padded, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));

    // Use wrappers for easy matrix access
    BwImage u(output);
    BwImage f(padded);

    // Translate original coordinates into coordinates for the padded matrix
    int x0 = (NLM_NAIVE_P+NLM_NAIVE_S)/2, xn = padded->width  - (NLM_NAIVE_P+NLM_NAIVE_S)/2,
        y0 = (NLM_NAIVE_P+NLM_NAIVE_S)/2, yn = padded->height - (NLM_NAIVE_P+NLM_NAIVE_S)/2,
        sx, sy;

    float f_x=0.0,
          c_x=0.0,
          tmp=0.0;

    // Iterate over padded matrix
    for(y0 = (NLM_NAIVE_P+NLM_NAIVE_S)/2; y0 < yn; ++y0)
    {
        for(x0 = (NLM_NAIVE_P+NLM_NAIVE_S)/2; x0 < xn; ++x0)
        {
            f_x = 0.0;
            c_x = 0.0;

            // Save patch around current pixel
            cvGetSubImage(padded, i_patch, cvRect(x0-NLM_NAIVE_P/2, y0-NLM_NAIVE_P/2, NLM_NAIVE_P, NLM_NAIVE_P));
            
            // Determine weighted distance for each patch as well as normalized constant
            for(sy = y0-NLM_NAIVE_S/2; sy < y0+NLM_NAIVE_S/2+1; ++sy)
            {
                for(sx = x0-NLM_NAIVE_S/2; sx < x0+NLM_NAIVE_S/2+1; ++sx)
                {
                    // Save patch around current iteration in the search window
                    cvGetSubImage(padded, j_patch, cvRect(sx-NLM_NAIVE_P/2, sy-NLM_NAIVE_P/2, NLM_NAIVE_P, NLM_NAIVE_P));

                    cvmSet(weight, sy+NLM_NAIVE_S/2-y0, sx+NLM_NAIVE_S/2-x0,
                        exp(-cvGaussianWeightedDistance(kernel, i_patch, j_patch)/
                        (NLM_NAIVE_H*NLM_NAIVE_H)));
                    c_x += cvmGet(weight, sy+NLM_NAIVE_S/2-y0, sx+NLM_NAIVE_S/2-x0);
                }
            }

            for(sy = y0-NLM_NAIVE_S/2; sy < y0+NLM_NAIVE_S/2+1; ++sy)
            {
                for(sx = x0-NLM_NAIVE_S/2; sx < x0+NLM_NAIVE_S/2+1; ++sx)
                {
                    tmp = cvmGet(weight, sy+NLM_NAIVE_S/2-y0, sx+NLM_NAIVE_S/2-x0) * f[sy][sx];
                    f_x += tmp;
                }
            }
            u[y0-(NLM_NAIVE_P+NLM_NAIVE_S)/2][x0-(NLM_NAIVE_P+NLM_NAIVE_S)/2] = (int)f_x/c_x;
        }
    }
}


/****************************************************************************
 * @brief  Calculate the sum of a given rectangle using the previously
 *         computed Integral Image matrix
 * @return The sum of the rectangle
 ****************************************************************************/
template<class T>
T si_sum(Image<T> img, int y, int x, int w)
{
    int d = w/2;
    x += d;
    y += d;
    double t = img[y+1][x+1] - img[y+1][x-w] - img[y-w][x+1] + img[y-w][x-w];
    //printf("%d=>%d@(%dx%d): %f-%f-%f+%f=%f\n",
    //w,d,x,y,img[y+1][x+1],img[y+1][x-w],img[y-w][x+1],img[y-w][x-w],t);
    return t;
}


/****************************************************************************
 * @brief Perform the Karnati/Uliyar version of the Non-Local Mean Method
 ****************************************************************************/
void nlm_mean(IplImage *original, IplImage *output)
{

    // Declare kernels
    static CvMat *kernel1 = cvGaussianKernel(NLM_MEAN_P1, NLM_MEAN_SIGMA);
    static CvMat *kernel2 = cvGaussianKernel(NLM_MEAN_P2, NLM_MEAN_SIGMA);
    static CvMat *kernel3 = cvGaussianKernel(NLM_MEAN_P3, NLM_MEAN_SIGMA);

    // Pad for largest kernel matrix
    static IplImage *padded = cvCreateImage(cvSize(original->width + NLM_MEAN_S + NLM_MEAN_P3,
                                                   original->height + NLM_MEAN_S + NLM_MEAN_P3),
                                            original->depth,
                                            original->nChannels);

    // Create weight matrix
    static CvMat *weight = cvCreateMat(NLM_MEAN_S, NLM_MEAN_S, CV_32FC1);

    // Center the original in the padded image
    CvPoint offset = cvPoint((NLM_MEAN_S + NLM_MEAN_P3 - 1)/2, (NLM_MEAN_S + NLM_MEAN_P3 - 1)/2);
    cvCopyMakeBorder(original, padded, offset, IPL_BORDER_CONSTANT, cvScalarAll(0));

    // Create N+1 x M+1 matrix for Integral Image(SI)
    IplImage *si_matrix = cvCreateImage(cvSize(padded->width+1,
                                               padded->height+1),
                                        IPL_DEPTH_64F,
                                        padded->nChannels);

    // Calculate Integral Image matrix
    cvIntegral(padded, si_matrix);

    // Set up easy matrix access
    BwImage u(output);
    BwImage o(padded);
    BwImageDouble s(si_matrix);

    // Average kernel matrices
    CvScalar _g1 = cvAvg(kernel1),
             _g2 = cvAvg(kernel2),
             _g3 = cvAvg(kernel3);

    double z,
           s1, s2, s3,
           g1 = _g1.val[0],
           g2 = _g2.val[0],
           g3 = _g3.val[0],
           f_x;

    //printf("g1 = %f; g2 = %f; g3 = %f\n", g1, g2, g3);

    int y0 = (NLM_MEAN_P3 + NLM_MEAN_S)/2,
        x0 = y0,

        yn = padded->height - (NLM_MEAN_P3 + NLM_MEAN_S)/2,
        xn = padded->width  - (NLM_MEAN_P3 + NLM_MEAN_S)/2,
        
        y, x;

    int sp,
        sy,
        sx,
        syn,
        sxn,
        
        p1,
        p2,
        p3;

    // Iterate over image
    for(y = y0; y < yn; ++y)
    {
        for(x = x0; x < xn; ++x)
        {
            sy  = y - NLM_MEAN_S/2;
            syn = y + NLM_MEAN_S/2 + 1;
            sxn = x + NLM_MEAN_S/2 + 1;

            // the sum of the weights for the current search window
            z = 0.0;

            // Compute rectangle for current neighborhood
            p1 = si_sum(s, y, x, NLM_MEAN_P1);
            p2 = si_sum(s, y, x, NLM_MEAN_P2);
            p3 = si_sum(s, y, x, NLM_MEAN_P3);

            // Iterate over search window
            for(;sy < syn; ++sy)
            {
                sx = x - NLM_MEAN_S/2;
                for(;sx < sxn; ++sx)
                {
                    sp = p1 - si_sum(s, sy, sx, NLM_MEAN_P1);
                    sp *= sp;
                    s1 = exp(-g1*sp/(NLM_MEAN_A1*NLM_MEAN_SIGMA*NLM_MEAN_SIGMA));

                    sp = p2 - si_sum(s, sy, sx, NLM_MEAN_P2);
                    sp *= sp;
                    s2 = exp(-g2*sp/(NLM_MEAN_A2*NLM_MEAN_SIGMA*NLM_MEAN_SIGMA));

                    sp = p3 - si_sum(s, sy, sx, NLM_MEAN_P3);
                    sp *= sp;
                    s3 = exp(-g3*sp/(NLM_MEAN_A3*NLM_MEAN_SIGMA*NLM_MEAN_SIGMA));

                    z += s1 + s2 + s3;
                    
                    cvmSet(weight, sy+NLM_MEAN_S/2-y, sx+NLM_MEAN_S/2-x, s1 + s2 + s3);
                }
            }

            f_x = 0.0;
            for(sy = y-NLM_MEAN_S/2; sy < y+NLM_MEAN_S/2+1; ++sy)
            {
                for(sx = x-NLM_MEAN_S/2; sx < x+NLM_MEAN_S/2+1; ++sx)
                {
                    f_x += cvmGet(weight, sy+NLM_MEAN_S/2-y, sx+NLM_MEAN_S/2-x) * o[sy][sx];
                }
            }

            u[y-(NLM_MEAN_P3+NLM_MEAN_S)/2][x-(NLM_MEAN_P3+NLM_MEAN_S)/2] = (int)f_x/z;
        }
    }
}


/****************************************************************************
 * @brief Add Gaussian Noise to an image with the given mean and variance.
 ****************************************************************************/
void addGaussianNoise(IplImage *image_in, IplImage *image_out, double mean, double var)
{
    double stddev = sqrt(var);
    static CvRNG rng_state = cvRNG(-1);

    cvRandArr(&rng_state, image_out, CV_RAND_NORMAL,
            cvRealScalar(mean*255), cvRealScalar(stddev*255));
    cvAdd(image_in, image_out, image_out);
}
