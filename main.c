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

#define ITERATIONS 5

#define WIN_MODIFIED "Modified"
#define WIN_ORIGINAL "Original"

using namespace std;

IplImage* process_image(IplImage *f, IplImage *u)
{
    /*
    // Let `u` be the current iteration
    IplImage *u = NULL,
             *f = NULL;

    f = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1);
    cvConvertImage(t, f, 0);
    
    // Convert to 8 bit grayscale
    u = cvCreateImage(cvSize(f->width, f->height),IPL_DEPTH_8U, 1);
    */

    cvCopy(f, u, NULL);

    non_convex(f, u, ITERATIONS);
    return u;
}

bool process_image_file(const string s1, const string s2)
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
    //cvReleaseImage(&t);

    bool save = !s2.empty();

    if(!(f || u))
    {
        return false;
    }

    process_image(f, u);

    if(save)
    {
        cout << "Saving image: `" << s2 << "`" << endl;
        cvSaveImage(s2.c_str(), u);
    }
    else
    {
        cout << "Show image" << endl;
        cvShowImage(WIN_MODIFIED, u);
        cvShowImage(WIN_ORIGINAL, f);
    }

    return true;
}

bool process_video_file(const string s1, const string s2)
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
                int isColor = 0;
                int frameH    = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
                int frameW    = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
                int fps       = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
                //int numFrames = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
                writer = cvCreateVideoWriter(s2.c_str(), CV_FOURCC('P','I','M','1'), fps,cvSize(frameW,frameH),isColor);
            }
            first = !first;
        }

        cvConvertImage(t, f, 0);
        process_image(f, u);

        if(save)
        {
            cvWriteFrame(writer, u);
        }
        else
        {
            cvWaitKey(20);
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

int main(int argc, char *argv[])
{
    string s1 = (argc > 1) ? argv[1] : "",
           s2 = (argc > 2) ? argv[2] : "";

    cout << "Argv[1]: " << s1 << endl << "Argv[2]: " << s2 << endl;

    cvNamedWindow(WIN_ORIGINAL, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(WIN_ORIGINAL, 0, 0);

    cvNamedWindow(WIN_MODIFIED, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(WIN_MODIFIED, 200, 200);

    cout << "Trying to process as image" << endl;
    if(!process_image_file(s1, s2))
    {
        cout << "Trying to process as video" << endl;
        if(!process_video_file(s1, s2))
        {
            print_options();
        }
    }

    /*
     * If s2 is empty, then the results are being
     * displayed in windows, so keep them open
     * until a key is pressed
     */
    if(s2.empty())
    {
        cout << "Waiting...." << endl;
        cvWaitKey(0);
    }

    return 0;
}

