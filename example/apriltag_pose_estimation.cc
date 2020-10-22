#include <iostream>
#include "opencv2/opencv.hpp"
#include <fcntl.h>
#include <errno.h> 
#include <cmath>
#include <cstring>
#include <vector>
#include <list>
#include <eigen3/Eigen/Dense>

extern "C" {
#include "apriltag_pose.h"
#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
#include "common/getopt.h"
#include "common/homography.h"
}
using namespace std;
using namespace cv;

#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;
/**
 * 定义了角度归一化函数，角度范围统一输出范围是[-pi,pi].
 **/
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}


    int main(int argc, char *argv[])
{
    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 1, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");

    if (!getopt_parse(getopt, argc, argv, 1) ||
            getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }
    // Initialize camera
    VideoCapture cap(1);
    if (!cap.isOpened()) {
        cerr << "Couldn't open video capture device" << endl;
        return -1;
    }
    // Initialize tag detector with options
    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11")) {
        tf = tag36h11_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tf = tagCircle21h7_create();
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tf = tagCircle49h12_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tf = tagStandard52h13_create();
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tf = tagCustom48h12_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");   //user does not set the family type of Tags 
        exit(-1);                                                           //default type:  tf = tag36h11_create()  
    }                   


    apriltag_detector_t *td = apriltag_detector_create();              //set some parameters
    apriltag_detector_add_family(td, tf);                              //set parttern recongnize of the specific tag family
    td->quad_decimate = getopt_get_double(getopt, "decimate");       
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");
    
    Mat frame, gray;
    
    /***********输入标定的相机参数*************/
    apriltag_detection_info_t info;     // parameters of the camera calibrations 在这里把标定得到的四元参数输入到程序里
     info.tagsize = 0.056; //标识的实际尺寸
     info.fx = 652.894;
     info.fy = 651.487;
     info.cx = 301.857;
     info.cy = 237.548;

    while (true) {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data , and turn the gray image into the candidate of the tags waiting to be recongnized(&im)
        image_u8_t im = { .width = gray.cols,        
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };                          

        zarray_t *detections = apriltag_detector_detect(td, &im);    //parttern recongnize to start
        cout << zarray_size(detections) << " tags detected" << endl;

        // Draw detection outlines
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;                       //
            zarray_get(detections, i, &det);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(0, 0xff, 0), 2);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 0, 0xff), 2);
            line(frame, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0xff, 0, 0), 2);
            line(frame, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0xff, 0, 0), 2);
            
            stringstream ss;
            ss << det->id;
            String text = ss.str();
            
         
           /*********加入位姿估计函数**********/
            info.det = det;  //det-->info
			apriltag_pose_t pose;
            estimate_pose_for_tag_homography(&info, &pose);
			//double err = estimate_tag_pose(&info, &pose);
         /**********将欧拉角转换成度为坐标，并归一化到[-pi,pi]范围内 ********/     
      double yaw = 180*standardRad(atan2(pose.R->data[3], pose.R->data[0]));
      double pitch=180*standardRad(sin(pose.R->data[6]));
      double roll=180*standardRad(atan2(pose.R->data[7],pose.R->data[8]));
             yaw=yaw/PI;
             pitch=pitch/PI;
             roll=roll/PI;    
			cout<<"THE 3D POSE:"<<"x= "<<pose.t->data[0]<<"  "<<"y= "<<pose.t->data[1]<<" "<<"z= "<<pose.t->data[2]<<endl;  //t output
            /************输出三维位置坐标信息***************/ 
                   cout<<"yaw: "<<yaw<<"'"<<endl;
                   cout<<"pitch: "<<pitch<<"'"<<endl;
                   cout<<"roll: "<<roll<<"'"<<endl;    
             /*************输出3个欧拉角数据****************/
        
		
		  int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
          double fontscale = 1.0;
          int baseline;
          Size textsize = getTextSize(text, fontface, fontscale, 2,
                                            &baseline);
            putText(frame, text, Point(det->c[0]-textsize.width/2,
                                       det->c[1]+textsize.height/2),
                    fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
        }
        zarray_destroy(detections);
        imshow("Tag Detections", frame);
        if (waitKey(25) >= 0)
            break;}
    apriltag_detector_destroy(td);

    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tagCircle21h7_destroy(tf);
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tagCircle49h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tagStandard52h13_destroy(tf);
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tagCustom48h12_destroy(tf);
    }
    getopt_destroy(getopt);
    return 0;
}
