#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#include <iostream>

// TCLAP
#include <algorithm>
#include <tclap/CmdLine.h>

#include <stdio.h>
#include <stdlib.h> // printf
#include <ctype.h>  // isdigit
#include "models.h"

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

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

using namespace TCLAP;
using namespace std;
using namespace cv;

typedef struct _options
{
    string method;
    int iterations;
    float var;
} options;


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
    //float norm=0.0;

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

    // Normalize kernel (seems to break output???)
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

void cvGetSubImage(IplImage* img, IplImage* subImg, CvRect roiRect)
{
    cvSetImageROI(img, roiRect);
    cvCopy(img, subImg, NULL);
    cvResetImageROI(img);
}

void nlm_naive(IplImage *original, IplImage *output)
{
    // Create a Guassian Kernel that is the same size as the patch
    static CvMat *kernel = cvGaussianKernel(NLM_NAIVE_P, NLM_NAIVE_SIGMA);

    // Create patches and search windows
    static IplImage *i_patch = cvCreateImage(cvSize(NLM_NAIVE_P, NLM_NAIVE_P), IPL_DEPTH_8U, 1),
                    *j_patch = cvCreateImage(cvSize(NLM_NAIVE_P, NLM_NAIVE_P), IPL_DEPTH_8U, 1);


    // NLM_NAIVE_Prepare for convolution
    static IplImage *padded = cvCreateImage(cvSize(original->width+NLM_NAIVE_S+NLM_NAIVE_P,
                                     original->height+NLM_NAIVE_S+NLM_NAIVE_P),
                                     original->depth,
                                     original->nChannels);
             

    static CvMat *weight = cvCreateMat(NLM_NAIVE_S, NLM_NAIVE_S, CV_32FC1);

    // Pad borders for convolution
    CvPoint offset = cvPoint((NLM_NAIVE_S+NLM_NAIVE_P-1)/2,(NLM_NAIVE_S+NLM_NAIVE_P-1)/2);
    cvCopyMakeBorder(original, padded, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));


    BwImage u(output);
    BwImage f(padded);

    int x0 = (NLM_NAIVE_P+NLM_NAIVE_S)/2, xn = padded->width  - (NLM_NAIVE_P+NLM_NAIVE_S)/2,
        y0 = (NLM_NAIVE_P+NLM_NAIVE_S)/2, yn = padded->height - (NLM_NAIVE_P+NLM_NAIVE_S)/2,
        sx, sy;

    float f_x=0.0,
          c_x=0.0,
          tmp=0.0;

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

/*
IplImage *si(IplImage *img)
{
    IplImage *ret = cvCreateImage(cvSize(img->width,
                                         img->height),
                                  IPL_DEPTH_64F,
                                  img->nChannels);

    int x,y;
    BwImage i(img);
    BwImageDouble j(ret);

    j[0][0] = i[0][0];

    for(x = 1; x < img->width; ++x)
    {
        j[0][x] = j[0][x-1] + i[0][x];
        //printf("j[0][%d]=j[0][%d]+(i[0][%d]=%d)=%f\n",x,x-1,x,i[0][x],j[0][x]);
    }

    for(y = 1; y < img->height; ++y)
    {
        j[y][0] = j[y-1][0] + i[y][0];
    }

    for(y = 1; y < img->height; ++y)
    {
        for(x = 1; x < img->width; ++x)
        {
            j[y][x] = j[y][x-1] + j[y-1][x] - j[y-1][x-1] + i[y][x];
            //printf("j[0][%d]=j[0][%d]+(i[0][%d]=%d)=%f\n",x,x-1,x,i[0][x],j[y][x]);
        }
    }

    return ret;
}
*/

template<class T>
T si_sum(Image<T> img, int y, int x, int w)
{
    int d = w/2;
    x += d;
    y += d;
    //printf("Dim: %d\n", d);
    double t = img[y+1][x+1] - img[y+1][x-w] - img[y-w][x+1] + img[y-w][x-w];
    //printf("%d=>%d@(%dx%d): %f-%f-%f+%f=%f\n",
    //w,d,x,y,img[y+1][x+1],img[y+1][x-w],img[y-w][x+1],img[y-w][x-w],t);

    //printf("(%dx%d):%f-%f-%f+%f=%f\n", y,x,img[y][x], img[y][x-w+1],
    //img[y-w+1][x], 2*img[y-w+1][x-w+1], t);
    return t;
}


void nlm_mean(IplImage *original, IplImage *output)
{
    //BwImage o(original);
    BwImage u(output);

    static CvMat *kernel1 = cvGaussianKernel(NLM_MEAN_P1, NLM_MEAN_SIGMA);
    static CvMat *kernel2 = cvGaussianKernel(NLM_MEAN_P2, NLM_MEAN_SIGMA);
    static CvMat *kernel3 = cvGaussianKernel(NLM_MEAN_P3, NLM_MEAN_SIGMA);

    // Pad for largest kernel matrix
    static IplImage *padded = cvCreateImage(cvSize(original->width  + NLM_MEAN_S + NLM_MEAN_P3,
                                                   original->height + NLM_MEAN_S + NLM_MEAN_P3),
                                            original->depth,
                                            original->nChannels);

    static CvMat *weight = cvCreateMat(NLM_MEAN_S, NLM_MEAN_S, CV_32FC1);

    // Center the original in the padded image
    CvPoint offset = cvPoint((NLM_MEAN_S + NLM_MEAN_P3 - 1)/2, (NLM_MEAN_S + NLM_MEAN_P3 - 1)/2);
    cvCopyMakeBorder(original, padded, offset, IPL_BORDER_CONSTANT, cvScalarAll(0));

    //IplImage *si_matrix = si(padded);
    BwImage o(padded);


    IplImage *si_matrix = cvCreateImage(cvSize(padded->width+1,
                                               padded->height+1),
                                        IPL_DEPTH_64F,
                                        padded->nChannels);
    cvIntegral(padded, si_matrix);
    BwImageDouble s(si_matrix);

    //BwImage p(padded);

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

    int sy,
        sx,

        syn,
        sxn;
        

    // Iterate over image
    for(y = (NLM_MEAN_P3 + NLM_MEAN_S)/2; y < padded->height - (NLM_MEAN_P3 + NLM_MEAN_S)/2; ++y)
    {
        for(x = (NLM_MEAN_P3 + NLM_MEAN_S)/2; x < padded->width - (NLM_MEAN_P3 + NLM_MEAN_S)/2; ++x)
        {

            sy = y - NLM_MEAN_S/2;

            syn = y + NLM_MEAN_S/2 + 1;
            sxn = x + NLM_MEAN_S/2 + 1;

            // the sum of the weights for the current search window
            z = 0.0;

            // Iterate over search window
            for(;sy < syn; ++sy)
            {
                sx = x - NLM_MEAN_S/2;
                for(;sx < sxn; ++sx)
                {
                    s1 = exp(-g1*pow(si_sum(s, y, x, NLM_MEAN_P1) -
                                    (si_sum(s, sy, sx, NLM_MEAN_P1)), 2)/(NLM_MEAN_A1*NLM_MEAN_SIGMA*NLM_MEAN_SIGMA));
                    s2 = exp(-g2*pow(si_sum(s, y, x, NLM_MEAN_P2) -
                                    (si_sum(s, sy, sx, NLM_MEAN_P2)), 2)/(NLM_MEAN_A2*NLM_MEAN_SIGMA*NLM_MEAN_SIGMA));
                    s3 = exp(-g3*pow(si_sum(s, y, x, NLM_MEAN_P3) -
                                    (si_sum(s, sy, sx, NLM_MEAN_P3)), 2)/(NLM_MEAN_A3*NLM_MEAN_SIGMA*NLM_MEAN_SIGMA));

                    z += s1 + s2 + s3;
                    
                    cvmSet(weight, sy+NLM_MEAN_S/2-y, sx+NLM_MEAN_S/2-x, s1 + s2 + s3);
                }
            }
            //printf("%dx%d\n", y, x);

            f_x = 0.0;
            for(sy = y-NLM_MEAN_S/2; sy < y+NLM_MEAN_S/2+1; ++sy)
            {
                for(sx = x-NLM_MEAN_S/2; sx < x+NLM_MEAN_S/2+1; ++sx)
                {
                    f_x += cvmGet(weight, sy+NLM_MEAN_S/2-y, sx+NLM_MEAN_S/2-x) * o[sy][sx];
                }
            }
            u[y-(NLM_MEAN_P3+NLM_MEAN_S)/2][x-(NLM_MEAN_P3+NLM_MEAN_S)/2] = (int)f_x/z;
            //printf("Setting:%d,%d = %f/%f = %d\n", y-(NLM_MEAN_P3+NLM_MEAN_S)/2,
            //x-(NLM_MEAN_P3+NLM_MEAN_S)/2, f_x, z, (int)f_x/z);
        }
    }




    //printf("j[0][0] = %d\n", s[0][0]);
    //int x, y;
    /*
    for(y = 0; y < NLM_MEAN_S+1; ++y)
    {
        for(x = 0; x < NLM_MEAN_S+1; ++x)
        {
            printf("%04d ", (int)s[y][x]);
        }
        cout << endl;
    }

    cout << endl << endl;

    for(y = 0; y < NLM_MEAN_S+1; ++y)
    {
        for(x = 0; x < NLM_MEAN_S+1; ++x)
        {
            printf("%04d ", p[y][x]);
        }
        cout << endl;
    }

    printf("%f\n", si_sum(s, 13, 13, 1));
    getchar();
    */


    //printf("o[-1][-1]: %f\n", s[si_matrix->height-1][si_matrix->width-1]);


}

void addGaussianNoise(IplImage *image_in, IplImage *image_out, double mean, double var)
{
    double stddev = sqrt(var);
    static CvRNG rng_state = cvRNG(-1);

    cvRandArr(&rng_state, image_out, CV_RAND_NORMAL,
            cvRealScalar(mean*255), cvRealScalar(stddev*255));
    cvAdd(image_in, image_out, image_out);
}

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

double psnr(IplImage *f_t, IplImage *u_t)
{
    BwImage f(f_t);
    BwImage u(u_t);

    int x,y,
        cols = f_t->width,
        rows = f_t->height;
    double numerator = log10(cols*rows)+log10(255*255),
           denominator = 0;

    for(x = 0; x < cols; ++x)
    {
        for(y = 0; y < rows; ++y)
        {
            denominator += pow(f[x][y] - u[x][y], 2);
        }
    }
    return 10*(numerator-log10(denominator));
}

void overlay_psnr(IplImage *f, IplImage *u)
{
    CvFont font;
    double hScale = 1;
    double vScale = 1;
    int lineWidth = 1;
    char buffer[35];
    sprintf(buffer, "PSNR:%f", psnr(f, u));

    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, hScale, vScale, 0, lineWidth);

    cvPutText (u, buffer, cvPoint(10, 20), &font, cvScalar(255, 255, 255));
}

IplImage* process_image(IplImage *f, IplImage *u, options opt)
{
    cvCopy(f, u, NULL);

    if(opt.method == "non-convex")
    {
        non_convex(f, u, opt.iterations);
    }
    else if(opt.method == "noise")
    {
        addGaussianNoise(f, u, 0, opt.var);
    }
    else if(opt.method == "nlm-naive")
    {
        nlm_naive(f, u);
    }
    else if(opt.method == "nlm-mean")
    {
        nlm_mean(f, u);
    }

    return u;
}

bool process_image_file(const string s1, const string s2, options opt)
{
    IplImage *t = cvLoadImage(s1.c_str(), -1);

    if(!t)
    {
        return false;
    }

    IplImage *f = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1),
             *u = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1);
    cvConvertImage(t, f, 0);

    bool save = !s2.empty();

    if(!(f || u))
    {
        return false;
    }

    process_image(f, u, opt);

    if(save)
    {
        printf("PSNR: %f\n", psnr(f, u));
        cvSaveImage(s2.c_str(), u);
    }
    else
    {
        overlay_psnr(f, u);
        cvShowImage(WIN_MODIFIED, u);
        cvShowImage(WIN_ORIGINAL, f);
    }

    return true;
}

bool process_video_file(const string s1, const string s2, options opt)
{
    CvCapture     *capture = NULL;
    CvVideoWriter *writer  = NULL;

    bool save = !s2.empty();

    if((s1.empty()) || isdigit(s1[0]))
    {
        int cam = atoi(s1.c_str());
        capture = cvCaptureFromCAM(cam);
    }
    else
    {
        capture = cvCaptureFromAVI(s1.c_str());
    }

    if(!capture)
    {
        return false;
    }

    IplImage *t = NULL,
             *p = NULL,
             *f = NULL,
             *u = NULL;

    bool first = true;
    while(1)
    {
        t = cvQueryFrame(capture);

        if(!t)
        {
            break;
        }

        if(first)
        {
            f = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1);
            u = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1);

            if(save)
            {
                p = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 3);
                int fps     = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS),
                    frameH  = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT),
                    frameW  = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);

                // Certain formats do not support slower frame rates
                // Ex: MPEG1/2 does not support 15/1 fps
                if(fps < 25)
                {
                    fps = 25;
                }
                writer = cvCreateVideoWriter(s2.c_str(), CV_FOURCC('P','I','M','1'), fps, cvSize(frameW,frameH));
            }
            first = !first;
        }

        cvConvertImage(t, f, 0);
        process_image(f, u, opt);

        cvWaitKey(20);
        if(save)
        {
            printf("PSNR: %f\n", psnr(f, u));

            // cvWriteFrame wants a 3 channel array even
            // though the values are grayscale.
            cvCvtColor(u, p, CV_GRAY2BGR);
            cvWriteFrame(writer, p);
        }
        else
        {
            overlay_psnr(f, u);
            cvShowImage(WIN_MODIFIED, u);
            cvShowImage(WIN_ORIGINAL, f);
        }
    }
    cvReleaseVideoWriter(&writer);
    cvReleaseCapture(&capture);

    return true;
}

int main(int argc, char **argv)
{
    string src,
           dst,
           method;
    float var;
    int itr;

    try
    {
        CmdLine cmd("Image Denoising Program", ' ', "0.1");

        UnlabeledValueArg<string> _src("source", "Source Image/Video", true, "", "string");
        cmd.add(_src);

        UnlabeledValueArg<string> _dst("destination", "Destination Image/Video", false, "", "string");
        cmd.add(_dst);

        ValueArg<float> _var("n", "variance", "Variance for zero-mean noise", false, 0.01, "float");
        cmd.add(_var);

        ValueArg<int> _itr("i", "iterations", "Iterations for non-convex method", false, 10, "int");
        cmd.add(_itr);

        // Limit methods to the following:
        vector<string> _allowed;
        _allowed.push_back("noise");
        _allowed.push_back("nlm-naive");
        _allowed.push_back("nlm-mean");
        _allowed.push_back("non-convex");
        //_allowed.push_back("nlm-conv");
        ValuesConstraint<string> _allowedMethods(_allowed);

        ValueArg<string> _method("m", "method", "Method to use on image/video", false, "noise", &_allowedMethods);
        cmd.add(_method);

        cmd.parse(argc, argv);

        // Saved the parsed command-line results into local variables
        src = _src.getValue();
        dst = _dst.getValue();

        itr = _itr.getValue();
        var = _var.getValue();

        method = _method.getValue();

        // Prepare the options that will be needed by the processing functions
        // This has only been used to keep the calling interfaces the same.
        options opt;
        opt.var = var;
        opt.method = method;
        opt.iterations = itr;

        /*
         * If a destination hasn't been specified,
         * show the results in windows.
         */
        if(dst.empty())
        {
            cvNamedWindow(WIN_ORIGINAL, CV_WINDOW_AUTOSIZE); 
            cvMoveWindow(WIN_ORIGINAL, 0, 0);

            cvNamedWindow(WIN_MODIFIED, CV_WINDOW_AUTOSIZE); 
            cvMoveWindow(WIN_MODIFIED, 200, 200);
        }

        if(!process_image_file(src, dst, opt))
        {
            puts("Failed to processes as an image. Assume video....");
            if(!process_video_file(src, dst, opt))
            {
                puts("Failed to process as video. File must be corrupt "
                     "or uses an unsupported format.");
            }
        }

        /*
         * If dst is empty, then the results are being
         * displayed in windows, so keep them open
         * until a key is pressed
         */
        if(dst.empty())
        {
            cout << "Press any key to exit...." << endl;
            cvWaitKey(0);
        }

	}
    catch (ArgException &e)  // catch any exceptions
	{
        cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
    }

    return 0;
}

