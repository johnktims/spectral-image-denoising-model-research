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

#include <stdlib.h> // printf
#include <ctype.h>  // isdigit
#include "models.h"

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

using namespace TCLAP;
using namespace std;
using namespace cv;


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

