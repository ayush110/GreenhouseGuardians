#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

int main() {
    try {
        fs::path save_dir = "./captures";
        fs::create_directories(save_dir);

        rs2::pipeline pipe;
        rs2::config cfg;

        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);

        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        float depth_scale = depth_sensor.get_depth_scale();
        std::cout << "Depth scale: " << depth_scale << " meters/unit\n";

        std::cout << "Press ENTER in terminal to capture.\n";
        std::cout << "Press q in preview window to quit.\n";

        // warm up
        for (int i = 0; i < 30; i++) {
            pipe.wait_for_frames();
        }

        while (true) {
            rs2::frameset frames = pipe.wait_for_frames();
            frames = align_to_color.process(frames);

            rs2::video_frame color_frame = frames.get_color_frame();
            rs2::depth_frame depth_frame = frames.get_depth_frame();

            if (!color_frame || !depth_frame) {
                continue;
            }

            cv::Mat color(
                cv::Size(color_frame.get_width(), color_frame.get_height()),
                CV_8UC3,
                (void*)color_frame.get_data(),
                cv::Mat::AUTO_STEP
            );

            cv::Mat depth(
                cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                CV_16UC1,
                (void*)depth_frame.get_data(),
                cv::Mat::AUTO_STEP
            );

            cv::Mat depth_8u, depth_colormap;
            depth.convertTo(depth_8u, CV_8U, 0.03);
            cv::applyColorMap(depth_8u, depth_colormap, cv::COLORMAP_JET);

            cv::Mat preview;
            cv::hconcat(color, depth_colormap, preview);

            int cx = depth.cols / 2;
            int cy = depth.rows / 2;
            float center_depth_m = depth_frame.get_distance(cx, cy);

            cv::putText(
                preview,
                "Center depth: " + std::to_string(center_depth_m) + " m",
                cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::imshow("RealSense Preview (Color | Depth)", preview);

            int key = cv::waitKey(1);
            if (key == 'q') {
                break;
            }

            if (std::cin.rdbuf()->in_avail() > 0) {
                std::string line;
                std::getline(std::cin, line);

                std::string ts = timestamp_now();

                fs::path color_path = save_dir / (ts + "_color.png");
                fs::path depth_path = save_dir / (ts + "_depth.png");
                fs::path depth_vis_path = save_dir / (ts + "_depth_vis.png");

                cv::imwrite(color_path.string(), color.clone());
                cv::imwrite(depth_path.string(), depth.clone()); // 16-bit raw depth
                cv::imwrite(depth_vis_path.string(), depth_colormap.clone());

                std::cout << "\nSaved:\n";
                std::cout << "  " << color_path << "\n";
                std::cout << "  " << depth_path << "\n";
                std::cout << "  " << depth_vis_path << "\n";
                std::cout << "  Center depth: " << center_depth_m << " m\n\n";
            }
        }

        pipe.stop();
        cv::destroyAllWindows();
    }
    catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/*
g++ capture_realsense.cpp -o capture_realsense \
    $(pkg-config --cflags --libs realsense2 opencv4) \
    -std=c++17  

./capture_realsense
*/