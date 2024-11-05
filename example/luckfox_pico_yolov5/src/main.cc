// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov5.h"
//#include "postprocess.h"
#include "agrd.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <unistd.h>   
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <time.h>
#include <vector>
#include <array>
using std::vector;
using std::array;
//opencv
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include "dma_alloc.cpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <iosfwd>

#define USE_DMA 0\

std::ostream& print_array(float* a, int size) {
    std::cout << "[";
    for (int i = 0; i < size; i++) {
        printf("%4.2f", a[i]);
        printf("_%4.2f", 10.05);
       // std::cout << a[i]; 
        if (i<size-1)
            std::cout << ", ";
    } 
    std::cout << "]";
    return std::cout;
}

std::ostream& print_array(float** a, int h, int w) {
    printf("MMMMMMMMMM %4.2f", a[0][0] );
    std::cout << "[";
    for (int i = 0; i < h; i++) {
        print_array( a[i], w); 
        if (i<w)
            std::cout << ", ";
    } 
    std::cout << "]";
    return std::cout;
}

void mapCoordinates(cv::Mat input, cv::Mat output, int *x, int *y) {	
	float scaleX = (float)output.cols / (float)input.cols; 
	float scaleY = (float)output.rows / (float)input.rows;
    
    *x = (int)((float)*x / scaleX);
    *y = (int)((float)*y / scaleY);
}
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
    return ((float)qnt - (float)zp) * scale; }
unsigned long long getTotalSystemMemory() {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return pages * page_size;
    }
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    // Check if ranges library is available 
    #ifdef __cpp_lib_ranges 
        std::cout << "Ranges library is Available.\n"; 
    #else 
        std::cout << "Ranges library is Not Available.\n"; 
    #endif
    #if __cplusplus >= 202002L // C++20 (and later) code
        printf("C++20")
    #endif
    //cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
    if (argc != 3)
    {
        printf("No modelpath, EXIT !\n");
        printf("%s <yolov5 model_path>\n", argv[0]);
        return -1;
    }
    system("RkLunch-stop.sh");
    const char *model_path = argv[1];
    const char *agrd_model_path = argv[2];

    clock_t start_time;
    clock_t end_time;
    char text[8];
    float fps = 0;

    //Model Input (Yolov5)
    //int model_width    = 640;
    //int model_height   = 640;
    int agrd_model_width    = 420;
    int agrd_model_height   = 300;
    int channels = 3;

    int ret;
    //rknn_app_context_t rknn_app_ctx;
    rknn_agrd_context_t rknn_agrd_app_ctx;
    object_detect_result_list od_results;
    init_post_process();
    
    start_time = clock();
    //memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
   // init_yolov5_model(model_path, &rknn_app_ctx);
    //release_yolov5_model(&rknn_app_ctx);
    
   // printf("------> Loading YO model requires %i ms\n", (clock()-start_time)/1000 );
    start_time = clock();
   // init_yolov5_model(model_path, &rknn_app_ctx);
   // printf("------> Loading model requires %i ms\n", (clock()-start_time)/1000 );
    
    printf("#-> Loading AGRD model requires %i ms MM :%d :\n", (clock()-start_time)/1000, getTotalSystemMemory() );
    
    start_time = clock(); 
    memset(&rknn_agrd_app_ctx, 0, sizeof(rknn_agrd_context_t));
    init_agrd_model(agrd_model_path, &rknn_agrd_app_ctx);
    printf("#-> Loading model requires %i ms MM :%d \n", 
    (clock()-start_time)/1000, getTotalSystemMemory() );
    bool DEBUG = false;
    int disp_flag = 0;
    int pixel_size = 0;
    size_t screensize = 0;
    int disp_width = 0;
    int disp_height = 0;
    void* framebuffer = NULL; 
    struct fb_fix_screeninfo fb_fix;
    struct fb_var_screeninfo fb_var;

    int framebuffer_fd = 0; //for DMA
    cv::Mat disp;

    int fb = open("/dev/fb0", O_RDWR); 
    fb = -1;
    if(fb == -1)
        printf("Screen OFF!\n");
    else 
        disp_flag = 1;
    if(disp_flag){
        ioctl(fb, FBIOGET_VSCREENINFO, &fb_var);
        ioctl(fb, FBIOGET_FSCREENINFO, &fb_fix);

        disp_width = fb_var.xres;
        disp_height = fb_var.yres;  
        pixel_size = fb_var.bits_per_pixel / 8;
        printf("Screen width = %d, Screen height = %d, Pixel_size = %d\n",disp_width, disp_height, pixel_size);
        
        screensize = disp_width * disp_height * pixel_size;
        framebuffer = (uint8_t*)mmap(NULL, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
        
        if( pixel_size == 4 )//ARGB8888
            disp = cv::Mat(disp_height, disp_width, CV_8UC3);
        else if ( pixel_size == 2 ) //RGB565
            disp = cv::Mat(disp_height, disp_width, CV_16UC1); 

#if USE_DMA
        dma_buf_alloc(RV1106_CMA_HEAP_PATH,
                      disp_width * disp_height * pixel_size,  
                      &framebuffer_fd, 
                      (void **) & (disp.data)); 
#endif
    }
    else{
        disp_width = 10;
        disp_height = 10;
    }
    
    cv::VideoCapture cap;
    //Init Opencv-mobile 
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  agrd_model_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, agrd_model_height);
    cap.set(cv::CAP_PROP_FPS, 8);
    cv::Mat bgr = cv::Mat(agrd_model_height, agrd_model_width, CV_8UC3);
    //cv::Mat frame_model_input(model_height, model_width, CV_8UC3,
    //       rknn_app_ctx.input_mems[0]->virt_addr);    
    cv::Mat agrd_frame_model_input(agrd_model_height, agrd_model_width, CV_8UC3,
     rknn_agrd_app_ctx.input_mems[0]->virt_addr);
    printf("---> line 4 \n");       
    bool cameraIsOpen = false;
    //cap.open("/dev/video11", cv::CAP_ANY); 
    cap.open(0);
    cv::Mat frame = cv::Mat(agrd_model_height, agrd_model_width, CV_8UC3); 
    for (int i=0;i<10;i++)
        cap >> bgr; 
    if(cap.isOpened()) {
        printf("---> Camera is open!\n");        
        cameraIsOpen = true;
        cap >> bgr; 
        //cv::cvtColor(bgr, frame, cv::COLOR_BGR2RGB);
        cv::imwrite("/root/ai/pic.jpg", bgr);
    } else {
        printf("---> Failed to open the camera !\n");
        exit(0);
    }
    printf("Starting the loop!\n");
    int k = 0;
    vector<cv::Mat> q; 
    int framesSpace = 2;
    int queueSize = nbrImgs*framesSpace;
    cv::Mat frames[queueSize];
    int size = sizeof(unsigned char)*agrd_model_width*agrd_model_height;             
    while(1) {
        cap >> bgr;
        if (q.size()>queueSize) {
            q.begin()->release();
            q.erase( q.begin() ); 
        }        
        if (false && k%4==0 && k<100) {
            //cv::cvtColor(bgr, frame, cv::COLOR_BGR2RGB);
            char* str;
            asprintf(&str, "/root/ai/pics/pic%d.jpg", k);
            bool check = cv::imwrite(str, bgr);
            free(str);
            if (check == false) {
                printf("save img fail \n");
            }
        }
        k +=1; 
        start_time = clock();
        //letterbox       
        if (DEBUG)
            printf("AGRD RESIZE\n");
        //cv::Mat agrd_frame = cv::Mat(agrd_model_height, agrd_model_height, CV_8UC3); 
        //cv::resize(bgr, agrd_frame, cv::Size(agrd_model_width,agrd_model_height), 
        //            0, 0, cv::INTER_LINEAR);
        //printf("AGRD BGR TO RGB\n");
        //cv::cvtColor(agrd_frame_model_input, agrd_frame, cv::COLOR_BGR2RGB);
        if (DEBUG)
            printf("AGRD PUSH TO QUEUE\n");
        q.push_back(bgr);

        //for(int i=0; i<bgr.rows; i++)
        //    for(int j=0; j<bgr.cols; j++) 
        //       bgr.ptr<uchar>(i)[j] = bgr.at<uchar>(i,j)/255
        
        if (q.size()>queueSize){
           if (DEBUG)
                printf("Calling AGRD... \n");
           for (int i=0;i<queueSize/framesSpace;i++) {              
              frames[i] = q[i*framesSpace];
              if (DEBUG)
                printf("copying ... size=%d \n", size);
              rknn_tensor_mem* input = rknn_agrd_app_ctx.input_mems[i];
              uchar* frm = frames[i].data;
              if (DEBUG)
                printf("copying now ... \n");  
              memcpy(input->virt_addr, frm, size);
            }
            inference_agrd_model(&rknn_agrd_app_ctx);
            uint8_t * output = (uint8_t *)(rknn_agrd_app_ctx.output_mems[0]->virt_addr); 
           
            int8_t k = output[0];
            float p1 = deqnt_affine_to_f32( k, rknn_agrd_app_ctx.output_attrs[0].zp, 
                                        rknn_agrd_app_ctx.output_attrs[0].scale);
            k = output[1];
            float p2 = deqnt_affine_to_f32( k, rknn_agrd_app_ctx.output_attrs[0].zp, 
                                        rknn_agrd_app_ctx.output_attrs[0].scale);
            end_time = clock();
            
            fps = ((float) CLOCKS_PER_SEC / (float)(end_time - start_time)) ;   
            printf("Mem %.2f MB FPS %.2f %f %% Agression VS %f %% No \n",
            getTotalSystemMemory()/(1.0*1024*1024), fps , p2, p1);
        }
        //inference_yolov5_model(&rknn_app_ctx, &od_results);
        //printf("od_results : length %d \n", od_results.count);
            
        // Add rectangle and probability
        if (false)
            for (int i = 0; i < od_results.count; i++) {
                //mapCoordinates(bgr, bgr_model_input, )
                object_detect_result *det_result = &(od_results.results[i]); 
                //mapCoordinates(bgr, bgr_model_input, &(det_result->box.left), &(det_result->box.left));
                //mapCoordinates(bgr, bgr_model_input, &det_result->box.right, &det_result->box.bottom);	
                printf(">>>> od_results %s \n", coco_cls_to_name(det_result->cls_id));
                printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                    det_result->box.left, det_result->box.top,
                    det_result->box.right, det_result->box.bottom,
                    det_result->prop);
                if (disp_flag) {
                    cv::rectangle(bgr,cv::Point(det_result->box.left ,det_result->box.top),
                                cv::Point(det_result->box.right,det_result->box.bottom),cv::Scalar(0,255,0),3);           
                    sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
                    cv::putText(bgr,text,cv::Point(det_result->box.left, det_result->box.top - 8),
                                            cv::FONT_HERSHEY_SIMPLEX,0.5,
                                            cv::Scalar(0,255,0),2);
                }  
            }
        if (od_results.count>0) {
            char* str;
            asprintf(&str, "./pics/detected_pic%i.jpg", k);
            cv::imwrite(str, bgr);
        }
        if(disp_flag) {
            //Fps Show
            sprintf(text,"fps=%.1f",fps); 
            cv::putText(bgr,text,cv::Point(0, 20),
                        cv::FONT_HERSHEY_SIMPLEX,0.5,
                        cv::Scalar(0,255,0),1);

            //LCD Show 
            if( pixel_size == 4 ) 
                cv::cvtColor(bgr, disp, cv::COLOR_BGR2BGRA);
            else if( pixel_size == 2 )
                cv::cvtColor(bgr, disp, cv::COLOR_BGR2BGR565);
            memcpy(framebuffer, disp.data, disp_width * disp_height * pixel_size);
#if USE_DMA
            dma_sync_cpu_to_device(framebuffer_fd);
#endif  
        }
        //Update Fps
        
        //printf("%s\n",text);
        memset(text,0,8); 
    }
   // deinit_post_process();

    if(disp_flag) {
        close(fb);
        munmap(framebuffer, screensize);
#if USE_DMA
        dma_buf_free(disp_width*disp_height*pixel_size,
                     &framebuffer_fd, 
                     bgr.data);
#endif
    }

   // ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov5_model fail! ret=%d\n", ret);
    }
    ret = release_agrd_model(&rknn_agrd_app_ctx);
    if (ret != 0)
    {
        printf("release_agrd_model fail! ret=%d\n", ret);
    }

    return 0;
}
