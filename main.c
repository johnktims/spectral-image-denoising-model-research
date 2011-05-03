#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#include <iostream>
#include <sstream> // isFloat

#include <stdio.h>  // printf
#include <ctype.h>  // isdigit
#include "models.h"

#define PI 3.14159265
#define ITERATIONS 5

#define S 21
#define P 7
#define H 10
#define SIGMA 6

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

using namespace std;
using namespace cv;

IplImage* si(IplImage *img)
{
    IplImage *ret = cvCreateImage(cvSize(img->width,
                                     img->height),
                                     img->depth,
                                     img->nChannels);

    int x,y;
    BwImage i(img);
    BwImage j(ret);

    j[0][0] = i[0][0];

    for(x = 1; x < img->width; ++x)
    {
        j[0][x] = j[0][x-1] + i[0][x];
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
        }
    }

    return ret;
}

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
    static CvMat *kernel = cvGaussianKernel(P, SIGMA);

    // Create patches and search windows
    static IplImage *i_patch  = cvCreateImage(cvSize(P, P), IPL_DEPTH_8U, 1),
             *j_patch  = cvCreateImage(cvSize(P, P), IPL_DEPTH_8U, 1);


    // Prepare for convolution
    static IplImage *padded = cvCreateImage(cvSize(original->width+S+P,
                                     original->height+S+P),
                                     original->depth,
                                     original->nChannels);
             

    static CvMat *weight = cvCreateMat(S, S, CV_32FC1);

    // Pad borders for convolution
    CvPoint offset = cvPoint((S+P-1)/2,(S+P-1)/2);
    cvCopyMakeBorder(original, padded, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));


    BwImage u(output);
    BwImage f(padded);

    int x0 = (P+S)/2, xn = padded->width  - (P+S)/2,
        y0 = (P+S)/2, yn = padded->height - (P+S)/2,
        sx, sy;

    float f_x=0.0,
          c_x=0.0,
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
}

float strToFloat(string s)
{
    std::istringstream iss(s);
    float f;
    iss >> noskipws >> f; // noskipws considers leading whitespace invalid
    return f;
}

bool isFloat(string s)
{
    std::istringstream iss(s);
    float f;
    iss >> noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail(); 
}

void addGaussianNoise(IplImage *image_in, IplImage *image_out, double mean, double var)
{
    double stddev = sqrt(var);
    CvRNG rng_state = cvRNG(-1);

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

IplImage* process_image(IplImage *f, IplImage *u, const string opt, const float var)
{
    cvCopy(f, u, NULL);

    if(opt == "convex")
    {
        non_convex(f, u, ITERATIONS);
    }
    else if(opt == "noise")
    {
        addGaussianNoise(f, u, 0, var);
    }
    else if(opt == "nlm")
    {
        nlm_naive(f, u);
    }

    return u;
}

bool process_image_file(const string s1, const string s2, const string opt, const float var)
{
    IplImage *t = cvLoadImage(s1.c_str(), -1);

    if(!t)
    {
        cout << "Failed to load image: " << s1 << endl;
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

    process_image(f, u, opt, var);

    if(save)
    {
        cout << "Saving image: `" << s2 << "`" << endl;
        cvSaveImage(s2.c_str(), u);
    }
    else
    {
        cout << "Show image" << endl;
        overlay_psnr(f, u);
        cvShowImage(WIN_MODIFIED, u);
        cvShowImage(WIN_ORIGINAL, f);
    }

    return true;
}

bool process_video_file(const string s1, const string s2, const string opt, const float var)
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
        process_image(f, u, opt, var);

        cvWaitKey(20);
        if(save)
        {
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

void print_options(void)
{
    puts("Syntax: ./program file.(jpg|png|pgm|etc)");
}

bool is_valid_option(const char* haystack[], const char* needle)
{
    const char** current = haystack;
    while(*current != NULL)
    {
        if(strcmp(needle, *current) == 0)
        {
            return true;
        }
        ++current;
    }
    return false;
}


int main(int argc, char *argv[])
{
    // Declare possible command-line options
    const char *options[] = {
        "noise",
        "convex",
        "nlm",
        NULL
    };

    // Set default parameters
    string src = "",
           dst = "",
           opt = "noise";

    float var = 0.01;

    // Process command-line options
    if(argc > 1)
    {
        if(!is_valid_option(options, argv[1]))
        {
            printf("`%s` is not a valid option. Defaulting to `noise`\n", argv[1]);
            src = argv[1];
            if(argc > 2)
            {
                if(!isFloat(argv[2]))
                {
                    dst = argv[2];
                    var = (argc > 3 && isFloat(argv[3]) && opt=="noise") ? strToFloat(argv[3]) : 0.01;
                }
                else
                {
                    var = strToFloat(argv[2]);
                }
            }
        }
        else
        {
            opt = argv[1];
            src = (argc > 2) ? argv[2] : "";
            if(argc > 3)
            {
                if(!isFloat(argv[3]))
                {
                    dst = argv[3];
                    var = (argc > 4 && isFloat(argv[4]) && opt=="noise") ? strToFloat(argv[4]) : 0.01;
                }
                else
                {
                    var = strToFloat(argv[2]);
                }
            }
        }
    }

    cout << "opt: " << opt << endl
         << "src: " << src << endl
         << "dst: " << dst << endl
         << "var: " << var << endl;

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

    if(!process_image_file(src, dst, opt, var))
    {
        if(!process_video_file(src, dst, opt, var))
        {
            print_options();
        }
    }

    /*
     * If dst is empty, then the results are being
     * displayed in windows, so keep them open
     * until a key is pressed
     */
    if(dst.empty())
    {
        cout << "Waiting...." << endl;
        cvWaitKey(0);
    }

    return 0;
}

