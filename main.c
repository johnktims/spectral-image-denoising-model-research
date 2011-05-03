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

#define ITERATIONS 5

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

using namespace std;

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

