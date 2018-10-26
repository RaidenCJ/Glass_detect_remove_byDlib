// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

#include <dlib/opencv.h>    
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        //人脸检测器
        frontal_face_detector face_detector = get_frontal_face_detector();
        //人脸特征提取器
        shape_predictor face_predictor;
        deserialize("shape_predictor_68_face_landmarks.dat") >> face_predictor;

        //glass检测器
        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        object_detector<image_scanner_type> glass_detector;
        deserialize("glass_detector.svm") >> glass_detector;
        
        //glass特征提取器
        shape_predictor glass_predictor;
        deserialize("glass_sp.dat") >> glass_predictor;
        
        std::vector<file> files = get_files_in_directory_tree(argv[2], match_ending(".jpg"));
        std::sort(files.begin(), files.end());
        if (files.size() == 0)
        {
            cout << "No images found in " << argv[2] << endl;
            return 1;
        }
        dlib::array2d<dlib::bgr_pixel> img;
        
        image_window win;
        // Loop over all the images provided on the command line.
        for (int i = 0; i < files.size(); ++i)
        {
            cout << "processing image " << files[i] << endl;

            load_image(img, files[i]);
                        
            // Make the image larger so we can detect small faces.
            pyramid_up(img);

            cv::Mat img1 = dlib::toMat(img);
            if(img1.empty()){
                cout <<"img1 is empty"<< endl;
                break;
            }
            
            //检测人脸
            std::vector<rectangle> face_rects = face_detector(img);
            cout << "Number of faces detected: " << face_rects.size() << endl;
            
            if(face_rects.size() == 0){    
                // Now let's view our face poses on the screen.
                win.clear_overlay();
                win.set_image(img);
                cout << "Hit enter to process the next image..." << endl;
                cin.get();  
                break;
            }
            
            //循环查找人脸
            for (unsigned long j = 0; j < face_rects.size(); ++j){
                //提取图片中的glass
                std::vector<rectangle> glass_rects = glass_detector(img);
                cout << "Number of glass detected: " << glass_rects.size() << endl;
                
                if(glass_rects.size() == 0){    
                    // Now let's view our face poses on the screen.
                    win.clear_overlay();
                    win.set_image(img);
                    cout << "Hit enter to process the next image..." << endl;
                    cin.get();  
                    break;
                }   
                
                //检测glass边缘
                std::vector<full_object_detection> glass_landmarks_shapes;
                full_object_detection glass_landmarks = glass_predictor(img, glass_rects[j]);
                cout << "glass number of parts: "<< glass_landmarks.num_parts() << endl;
                cout << "glass pixel position of first part:  " << glass_landmarks.part(0) << endl;
                cout << "glass pixel position of second part: " << glass_landmarks.part(1) << endl;
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                glass_landmarks_shapes.push_back(glass_landmarks);  

                //提取人脸的轮廓
                std::vector<full_object_detection> face_landmarks_shapes;
                full_object_detection face_landmarks = face_predictor(img, face_rects[j]);
                cout << "face number of parts: "<< face_landmarks.num_parts() << endl;
                cout << "face pixel position of first part:  " << face_landmarks.part(0) << endl;
                cout << "face pixel position of second part: " << face_landmarks.part(1) << endl;
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                face_landmarks_shapes.push_back(face_landmarks);                    
                
                //for (unsigned long k = 0; k < face_landmarks.num_parts(); k++) 
                //{
                //    dlib::point p = face_landmarks.part(k);
                //    cout <<"index:" << k <<"    face_landmarks x:"<< p.x()<<"   face_landmarks  y:"<<p.y() << endl;
                //    // p 点的直径 3 参数为原点直径 rgb_pixel 颜色
                //    cv::circle(img1,cv::Point(p.x(),p.y()),3,CV_RGB(0,255,0));
                //}
                //
                //for (unsigned long k = 0; k < glass_landmarks.num_parts(); k++) 
                //{
                //    dlib::point p = glass_landmarks.part(k);
                //    cout <<"index:" << k <<"    glass_landmarks x:"<< p.x()<<"   glass_landmarks  y:"<<p.y() << endl;
                //    // p 点的直径 3 参数为原点直径 rgb_pixel 颜色
                //    //cv::circle(img1,cv::Point(p.x(),p.y()),3,CV_RGB(0,0,255));
                //}
                //
                ////画下巴
                //for (unsigned long k = 0; k < 2; k++){
                //    dlib::point p = face_landmarks.part(k);
                //    dlib::point p1 = face_landmarks.part(k+1);
                //    //cv::line(img1,cv::Point(p.x(),p.y()),cv::Point(p1.x(),p1.y()),(0,255,0),5);
                //}
                //    
                //for (unsigned long k = 14; k < 16; k++){
                //    dlib::point p = face_landmarks.part(k);
                //    dlib::point p1 = face_landmarks.part(k+1);
                //    //cv::line(img1,cv::Point(p.x(),p.y()),cv::Point(p1.x(),p1.y()),(0,255,0),5);
                //}
                ////画左眉毛
                //for (unsigned long k = 17; k < 21; k++){
                //    dlib::point p = face_landmarks.part(k);
                //    dlib::point p1 = face_landmarks.part(k+1);
                //    //cv::line(img1,cv::Point(p.x(),p.y()),cv::Point(p1.x(),p1.y()),(0,255,0),5);
                //}
                ////画右眉毛
                //for (unsigned long k = 22; k < 26; k++){
                //    dlib::point p = face_landmarks.part(k);
                //    dlib::point p1 = face_landmarks.part(k+1);
                //    //cv::line(img1,cv::Point(p.x(),p.y()),cv::Point(p1.x(),p1.y()),(0,255,0),5);
                //}
                ////画鼻子
                ////cv::line(img1,cv::Point(face_landmarks.part(30).x(),face_landmarks.part(30).y()),cv::Point(face_landmarks.part(31).x(),face_landmarks.part(31).y()),(0,255,0),5);
                ////cv::line(img1,cv::Point(face_landmarks.part(30).x(),face_landmarks.part(30).y()),cv::Point(face_landmarks.part(35).x(),face_landmarks.part(35).y()),(0,255,0),5);
                //
                ////画左眼
                //for (unsigned long k = 36; k < 41; k++){
                //    dlib::point p = face_landmarks.part(k);
                //    dlib::point p1 = face_landmarks.part(k+1);
                //    //cv::line(img1,cv::Point(p.x(),p.y()),cv::Point(p1.x(),p1.y()),(0,255,0),5);
                //}
                ////cv::line(img1,cv::Point(face_landmarks.part(36).x(),face_landmarks.part(36).y()),cv::Point(face_landmarks.part(41).x(),face_landmarks.part(41).y()),(0,255,0),5);
                //
                ////画右眼
                //for (unsigned long k = 42; k < 47; k++){
                //    dlib::point p = face_landmarks.part(k);
                //    dlib::point p1 = face_landmarks.part(k+1);
                //    //cv::line(img1,cv::Point(p.x(),p.y()),cv::Point(p1.x(),p1.y()),(0,255,0),5);
                //}
                //cv::line(img1,cv::Point(face_landmarks.part(42).x(),face_landmarks.part(42).y()),cv::Point(face_landmarks.part(47).x(),face_landmarks.part(47).y()),(0,255,0),5);
                //cv::line(img1,cv::Point(face_landmarks.part(21).x(),face_landmarks.part(21).y()),cv::Point(face_landmarks.part(22).x(),face_landmarks.part(22).y()),(0,255,0),5);
                //cv::line(img1,cv::Point(face_landmarks.part(35).x(),face_landmarks.part(35).y()),cv::Point(face_landmarks.part(14).x(),face_landmarks.part(14).y()),(0,255,0),5);
                //cv::line(img1,cv::Point(face_landmarks.part(0).x(),face_landmarks.part(0).y()),cv::Point(face_landmarks.part(17).x(),face_landmarks.part(17).y()),(0,255,0),5);
                //cv::line(img1,cv::Point(face_landmarks.part(16).x(),face_landmarks.part(16).y()),cv::Point(face_landmarks.part(26).x(),face_landmarks.part(26).y()),(0,255,0),5);
                //cv::line(img1,cv::Point(face_landmarks.part(2).x(),face_landmarks.part(2).y()),cv::Point(face_landmarks.part(31).x(),face_landmarks.part(31).y()),(0,255,0),5);

                //dlib::array2d<dlib::bgr_pixel> img1_im;
                //image_window win_img1_im;
                //dlib::assign_image(img1_im, dlib::cv_image<bgr_pixel>(img1));
                //win_img1_im.set_image(img1_im);                
                
                
                ///////////////////////////////////////////////////////////////////////////////////////////////找人戴眼镜区域
                cv::Mat face_glass_Img;
                cv::Mat mask = cv::Mat::zeros(img1.size(),CV_8UC3); 
                std::vector<std::vector< cv::Point>> contour;  
                std::vector<cv::Point> pts;  
                //0 1 2
                
                //画下巴
                pts.push_back(cv::Point(face_landmarks.part(0).x(),face_landmarks.part(17).y()));
                for (unsigned long k = 0; k <= 2; k++){
                    dlib::point p = face_landmarks.part(k);
                    pts.push_back(cv::Point(p.x()+10,p.y()));
                }
                pts.push_back(cv::Point(face_landmarks.part(31).x(),face_landmarks.part(31).y()));
                pts.push_back(cv::Point(face_landmarks.part(29).x(),face_landmarks.part(29).y()));
                pts.push_back(cv::Point(face_landmarks.part(35).x(),face_landmarks.part(35).y()));
                
                for (unsigned long k = 14; k <=16; k++){
                    dlib::point p = face_landmarks.part(k);
                    pts.push_back(cv::Point(p.x()-10,p.y()));
                }
                pts.push_back(cv::Point(face_landmarks.part(16).x(),face_landmarks.part(26).y()));
                //画右眉毛
                //int offset_r = (face_landmarks.part(26).y() - face_landmarks.part(25).y())/2;
                //for (unsigned long k = 26; k >= 22; k--){
                //    dlib::point p = face_landmarks.part(k);
                //    pts.push_back(cv::Point(p.x(),p.y()+offset_r));
                //}  
                pts.push_back(cv::Point(face_landmarks.part(26).x(),face_landmarks.part(26).y()));
                pts.push_back(cv::Point(face_landmarks.part(22).x(),face_landmarks.part(22).y()));
                //画左眉毛
                //int offset_l = (face_landmarks.part(17).y() - face_landmarks.part(18).y())/2;
                //for (unsigned long k = 21; k >= 17; k--){
                //    dlib::point p = face_landmarks.part(k);
                //    pts.push_back(cv::Point(p.x(),p.y()+offset_l));//offset(point18-point17)
                //}
                pts.push_back(cv::Point(face_landmarks.part(21).x(),face_landmarks.part(21).y()));
                pts.push_back(cv::Point(face_landmarks.part(17).x(),face_landmarks.part(17).y()));
                
                contour.push_back(pts); 
                cv::drawContours(mask,contour,0,cv::Scalar::all(255),-1);  
                img1.copyTo(face_glass_Img,mask);
                
                //dlib::array2d<dlib::bgr_pixel> dstImg_im;
                //image_window win_dstImg;
                //dlib::assign_image(dstImg_im, dlib::cv_image<bgr_pixel>(face_glass_Img));
                //win_dstImg.set_image(dstImg_im);                
                //////////////////////////////////////////////////////////////////////////////////////
                
                //////////////////////////////////////////////////////////////////////////找眼睛区域
                cv::Mat eye_img_l;
                cv::Mat mask2 = cv::Mat::zeros(img1.size(),CV_8UC3); 
                std::vector<std::vector< cv::Point>> contour2;  
                std::vector<cv::Point> pts2;  
                
                //画左眼
                for (unsigned long k = 36; k <= 41; k++){
                    dlib::point p = face_landmarks.part(k);
                    pts2.push_back(cv::Point(p.x(),p.y()));
                }
                
                contour2.push_back(pts2); 
                cv::drawContours(mask2,contour2,0,cv::Scalar::all(255),-1);  
                img1.copyTo(eye_img_l,mask2); 
                
                //dlib::array2d<dlib::bgr_pixel> dstImg2_im;
                //image_window win_dstImg2;
                //dlib::assign_image(dstImg2_im, dlib::cv_image<bgr_pixel>(eye_img_l));
                //win_dstImg2.set_image(dstImg2_im);   
                
                
                cv::Mat eye_img_r;
                cv::Mat mask3 = cv::Mat::zeros(img1.size(),CV_8UC3); 
                std::vector<std::vector< cv::Point>> contour3;  
                std::vector<cv::Point> pts3;  
                //画右眼
                for (unsigned long k = 42; k <= 47; k++){
                    dlib::point p = face_landmarks.part(k);
                    pts3.push_back(cv::Point(p.x(),p.y()));
                }
                contour3.push_back(pts3); 
                cv::drawContours(mask3,contour3,0,cv::Scalar::all(255),-1);  
                img1.copyTo(eye_img_r,mask3); 
                
                //dlib::array2d<dlib::bgr_pixel> dstImg2_im;
                //image_window win_dstImg2;
                //dlib::assign_image(dstImg2_im, dlib::cv_image<bgr_pixel>(eye_img_r));
                //win_dstImg2.set_image(dstImg2_im);   
                ////////////////////////////////////////////////////////////////////////////////
                
                
                ////////////////////////////////////////////////////////////////////////////////////////找到眼镜所在区域
                dlib::array2d<dlib::bgr_pixel> win_result_Img_im;
                cv::Mat glass_Img = face_glass_Img - eye_img_l - eye_img_r;
                
                cv::Mat new_Img, new_Img_ok;
                new_Img = img1 - face_glass_Img;
                
                //填充眼镜
                dlib::array2d<dlib::bgr_pixel> glass_Img_im1;
                image_window win_glass_Img1;
                dlib::assign_image(glass_Img_im1, dlib::cv_image<bgr_pixel>(glass_Img));
                win_glass_Img1.set_image(glass_Img_im1);
                //找30的点
                double full_B =0;
                double full_G =0;
                double full_R =0;
                int pix_count =0;
                dlib::point point_cer = face_landmarks.part(30);
                std::cout<<"glass_Img[30].x:"<<point_cer.x()<<"  " <<"glass_Img[30].y:"<<point_cer.y()<<std::endl;
                ////////////////////////////////
                //    A   C   B
                //     ...D
                ////////////////////////////////
                int pointA = face_landmarks.part(11).x();//w1
                int pointB = face_landmarks.part(13).x();//w2
                int pointC = face_landmarks.part(14).y();//h1
                int pointD = face_landmarks.part(13).y();//h2
                
                std::cout<<"3.x " <<face_landmarks.part(11).x()/2<<  "     3.y "<<face_landmarks.part(11).y()/2<<std::endl;
                std::cout<<"13.x " <<face_landmarks.part(13).x()/2<<  "   13.y "<<face_landmarks.part(13).y()/2<<std::endl;
                std::cout<<"29.x " <<face_landmarks.part(14).x()/2<<  "   29.y "<<face_landmarks.part(14).y()/2<<std::endl;
                std::cout<<"30.x " <<face_landmarks.part(13).x()/2<<  "   33.y "<<face_landmarks.part(13).y()/2<<std::endl;
                
                for(int i=0;i<img1.rows;i++)
                {
                    for(int j=0;j<img1.cols;j++)
                    {
                        //取一块区域
                        if(i >=pointA && i <=pointB && j>=pointC && j<=pointD)
                        {
                            cv::Vec3b& pix_rgba = img1.at<cv::Vec3b>(i, j); 
                            full_B += pix_rgba[0];
                            full_G += pix_rgba[1];
                            full_R += pix_rgba[2];
                            std::cout<<"(x,y):"<<i/2<<"-"<<j/2 <<"  count: "<<pix_count<<"  full_B:"<<(int)pix_rgba[0]<<"  " <<"full_G:"<<(int)pix_rgba[1]<<"  "<<"full_R:"<<(int)pix_rgba[2]<<"  "<<std::endl;
                            //std::cout<<full_B<<"  " <<"full_G:"<<full_G<<"  "<<"full_R:"<<full_R<<"  "<<std::endl;
                            pix_count++;
                        }
                    }
                }
                full_B /= pix_count;
                full_G /= pix_count;
                full_R /= pix_count;
                
                std::cout<<"result " <<full_B<<"  " <<"full_G:"<<full_G<<"  "<<"full_R:"<<full_R<<"  "<<std::endl;
                
                for (int i = 0; i < glass_Img.rows; i++)
                {
                    for (int j = 0; j < glass_Img.cols; j++)
                    {
                        cv::Vec3b& pix_rgba = glass_Img.at<cv::Vec3b>(i, j);
                        if(pix_rgba[0]!=0 && pix_rgba[1]!=0 && pix_rgba[2]!=0){
                            if(pix_rgba[0]<full_B && pix_rgba[1]<full_G && pix_rgba[2]<full_R)
                            {
                                pix_rgba[0] = full_B;//B
                                pix_rgba[1] = full_G;//G
                                pix_rgba[2] = full_R;//R
                            }
                        }
                    }
                }
                
                dlib::array2d<dlib::bgr_pixel> glass_Img_im;
                image_window win_glass_Img;
                dlib::assign_image(glass_Img_im, dlib::cv_image<bgr_pixel>(glass_Img));
                win_glass_Img.set_image(glass_Img_im);
                
                
                dlib::array2d<dlib::bgr_pixel> new_Img_im1;
                image_window win_new_Img_im1;
                dlib::assign_image(new_Img_im1, dlib::cv_image<bgr_pixel>(new_Img));
                win_new_Img_im1.set_image(new_Img_im1);
                
                new_Img_ok = new_Img + glass_Img + eye_img_l + eye_img_r;
                image_window win_result_Img;
                dlib::assign_image(win_result_Img_im, dlib::cv_image<bgr_pixel>(new_Img_ok));
                win_result_Img.set_image(win_result_Img_im);
                ///////////////////////////////////////////////////////////////////////////////////////////
                
                // Now let's view our face poses on the screen.                
                win.clear_overlay();
                win.set_image(img);
                win.add_overlay(face_rects, rgb_pixel(255,0,0));//glass矩形位置
                win.add_overlay(glass_rects, rgb_pixel(0,255,0));
                
                save_jpeg(win_result_Img_im, "OK.jpeg",100);
                
                // We can also extract copies of each face that are cropped, rotated upright,
                // and scaled to a standard size as shown here:
                //image_window win_faces;
                //dlib::array<array2d<rgb_pixel> > face_chips;
                //extract_image_chips(img, get_face_chip_details(face_landmarks_shapes), face_chips);
                //win_faces.set_image(tile_images(face_chips));
                
                //win.add_overlay(render_face_detections(face_landmarks_shapes));
                cout << "Hit enter to process the next image..." << endl;
                cin.get();
                ;
            }
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

