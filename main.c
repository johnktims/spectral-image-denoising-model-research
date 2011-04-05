/**
 * compile : gcc main.c -O3 -lm
 * make gif: convert -delay 10 -loop 0 Iteration_*pgm out.gif
 */

#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#endif

#include <stdio.h>  // printf
#include "models.h"

int main(int argc, char *argv[])
{
    int x, y, i, I = 5;

    CvCapture* capture = NULL;


    // Load noisy pgm `f` with Neumann B.C.s
    IplImage *f = NULL,
             *t = NULL,
    
    // Let `u` be the original pgm
             *u = NULL;

    
    // load the passed in pgm or try to open
    // the lena pgm in this directory
    /*
    if(argc > 1)
    {
        capture = cvCaptureFromAVI(argv[1]);

        if(!capture)
        {
            puts("NOT CAPTURING");
            t = cvLoadImage(argv[1], -1);
            f = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1);
            cvConvertImage(t, f, 0);
        }else
        {
            puts("CAPTURING");
        }
    }
    */

    /*
    if(!f || !capture)
    {
        puts("Invalid image file.");
        puts("Syntax: ./program file.(jpg|png|pgm|etc)");
        return 0;
    }
    */
    capture = cvCaptureFromAVI(argv[1]);

    //printf("Image dimensions: %d by %d with %d Channels\n", f->width, f->height, f->nChannels);

    cvNamedWindow("Modified", CV_WINDOW_AUTOSIZE); 
    cvMoveWindow("Modified", 100, 100);

    cvNamedWindow("Original", CV_WINDOW_AUTOSIZE); 
    cvMoveWindow("Original", 200, 200);

    t = cvQueryFrame(capture);

    int isColor = 0;
    int frameH    = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    int frameW    = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    int fps       = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
    int numFrames = (int) cvGetCaptureProperty(capture,  CV_CAP_PROP_FRAME_COUNT);
    int index = 0;

    CvVideoWriter *writer = cvCreateVideoWriter("out.avi", CV_FOURCC('P','I','M','1'), fps,cvSize(frameW,frameH),isColor);

    if(t)
    {
        printf("Video: %dx%d @ %d; %d frames\n", frameH, frameW, fps, numFrames);
        f = cvCreateImage(cvSize(t->width, t->height),IPL_DEPTH_8U, 1);
    }
    while(1)
    {
        if(!t)
        {
            break;
        }

        if(index % 10 == 0)
        {
            cvConvertImage(t, f, 0);
            u = cvCloneImage(f);

            cvShowImage("Original", f);

            non_convex(f, u, I);
            cvShowImage("Modified", u);
            cvWaitKey(20);
            cvWriteFrame(writer, u);
        }

        t = cvQueryFrame(capture);
        ++index;
    }

    cvReleaseVideoWriter(&writer);
    cvReleaseCapture(&capture);
    cvReleaseImage(&f);
    cvReleaseImage(&u);
    return 0;
}

