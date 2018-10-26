// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how you can use dlib to make an object detector
    for things like faces, pedestrians, and any other semi-rigid object.  In
    particular, we go though the steps to train the kind of sliding window
    object detector first published by Dalal and Triggs in 2005 in the paper
    Histograms of Oriented Gradients for Human Detection.  

    Note that this program executes fastest when compiled with at least SSE2
    instructions enabled.  So if you are using a PC with an Intel or AMD chip
    then you should enable at least SSE2 instructions.  If you are using cmake
    to compile this program you can enable them by using one of the following
    commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  

*/


#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>

#include <dlib/image_io.h>
#include <dlib/dir_nav.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  

    try
    {
        // In this example we are going to train a face detector based on the
        // small faces dataset in the examples/faces directory.  So the first
        // thing we do is load that dataset.  This means you need to supply the
        // path to this faces folder as a command line argument so we will know
        // where it is.
        if (argc != 2)
        {
            cout << "Give the path to the examples/faces directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./fhog_object_detector_test faces" << endl;
            cout << endl;
            return 0;
        }
        
        // Get the list of video frames.  
        std::vector<file> files = get_files_in_directory_tree(argv[1], match_ending(".jpg"));
        std::sort(files.begin(), files.end());
        if (files.size() == 0)
        {
            cout << "No images found in " << argv[1] << endl;
            return 1;
        }

        dlib::array2d<dlib::rgb_pixel> img_rgb;

        // Then you can recall it using the deserialize() function.
        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        object_detector<image_scanner_type> detector2;
        deserialize("glass_detector.svm") >> detector2;
        
        image_window hogwin(draw_fhog(detector2), "Learned fHOG detector");
        
        // Now run the tracker.  All we have to do is call tracker.update() and it will keep
        // track of the juice box!
        image_window win;
        for (unsigned long i = 0; i < files.size(); ++i)
        {
            load_image(img_rgb, files[i]);
            
            // Run the detector and get the face detections.
            std::vector<rectangle> dets = detector2(img_rgb);
            if(dets.size() == 0){
                //没有眼镜
            }
            win.clear_overlay();
            win.set_image(img_rgb);
            win.add_overlay(dets, rgb_pixel(255,0,0));
            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

