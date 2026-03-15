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
        fs::path base_dir = fs::absolute("./captures");

        fs::path rgb_dir = base_dir / "rgb";
        fs::path depth_dir = base_dir / "depth";
        fs::path depth_vis_dir = base_dir / "depth_vis";

        fs::create_directories(rgb_dir);
        fs::create_directories(depth_dir);
        fs::create_directories(depth_vis_dir);

        std::cout << "Saving RGB to: " << rgb_dir << "\n";
        std::cout << "Saving depth to: " << depth_dir << "\n";
        std::cout << "Saving depth visualization to: " << depth_vis_dir << "\n";

        rs2::pipeline pipe;
        rs2::config cfg;

        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);

        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        float depth_scale = depth_sensor.get_depth_scale();

        std::cout << "Depth scale: " << depth_scale << " meters/unit\n";
        std::cout << "Press s in preview window to save\n";
        std::cout << "Press q in preview window to quit\n";

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

            cv::Mat color_copy = color.clone();
            cv::Mat depth_copy = depth.clone();

            cv::Mat depth_meters;
            depth_copy.convertTo(depth_meters, CV_32F, depth_scale);

            cv::Mat valid_mask = depth_copy > 0;

            float focus_min = 0.3f;
            float focus_max = 1.5f;

            cv::Mat vis_depth = depth_meters.clone();
            vis_depth.setTo(focus_min, vis_depth < focus_min);
            vis_depth.setTo(focus_max, vis_depth > focus_max);

            cv::Mat depth_8u;
            vis_depth.convertTo(
                depth_8u,
                CV_8U,
                255.0 / (focus_max - focus_min),
                -focus_min * 255.0 / (focus_max - focus_min)
            );

            cv::Mat equalized;
            cv::equalizeHist(depth_8u, equalized);

            cv::Mat invalid_mask;
            cv::bitwise_not(valid_mask, invalid_mask);
            equalized.setTo(0, invalid_mask);

            cv::Mat depth_colormap;
            cv::applyColorMap(equalized, depth_colormap, cv::COLORMAP_TURBO);

            cv::Mat preview;
            cv::hconcat(color_copy, depth_colormap, preview);

            int cx = depth_copy.cols / 2;
            int cy = depth_copy.rows / 2;
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

            cv::putText(
                preview,
                "Press s to save, q to quit",
                cv::Point(20, 65),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::imshow("RealSense Preview (Color | Depth)", preview);

            int key = cv::waitKey(1) & 0xFF;

            if (key == 'q') {
                break;
            }

            if (key == 's') {
                std::string ts = timestamp_now();

                fs::path color_path = rgb_dir / (ts + ".png");
                fs::path depth_path = depth_dir / (ts + ".png");
                fs::path depth_vis_path = depth_vis_dir / (ts + ".png");

                bool ok1 = cv::imwrite(color_path.string(), color_copy);
                bool ok2 = cv::imwrite(depth_path.string(), depth_copy);
                bool ok3 = cv::imwrite(depth_vis_path.string(), depth_colormap);

                std::cout << "\nSaved:\n";
                std::cout << "  RGB:       " << color_path << "\n";
                std::cout << "  Depth:     " << depth_path << "\n";
                std::cout << "  Depth vis: " << depth_vis_path << "\n";
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