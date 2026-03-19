#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <thread>
#include <ctime>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct SharedFrames {
    cv::Mat rgb_bgr;
    cv::Mat depth_raw_16u;
    cv::Mat depth_color_bgr;
    std::string timestamp;
    uint64_t frame_counter = 0;
};

static SharedFrames g_frames;
static std::mutex g_mutex;
static std::condition_variable g_cv;
static std::atomic<bool> g_running{true};

static std::mutex g_meta_mutex;
static float g_depth_scale = 0.0f;
static rs2_intrinsics g_color_intrinsics{};
static bool g_intrinsics_ready = false;
static std::atomic<bool> g_camera_connected{false};

uint64_t now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
}

void wait_until_trigger_ms(uint64_t trigger_at_ms) {
    while (true) {
        uint64_t current = now_ms();
        if (current >= trigger_at_ms) return;

        uint64_t remaining = trigger_at_ms - current;
        if (remaining > 20) {
            std::this_thread::sleep_for(std::chrono::milliseconds(remaining - 10));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;

    std::tm tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S")
        << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::vector<uchar> encode_jpeg(const cv::Mat& img, int quality = 80) {
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", img, buf, params);
    return buf;
}

float get_depth_scale_from_profile(const rs2::pipeline_profile& profile) {
    rs2::device dev = profile.get_device();
    for (rs2::sensor s : dev.query_sensors()) {
        if (auto ds = s.as<rs2::depth_sensor>()) {
            return ds.get_depth_scale();
        }
    }
    throw std::runtime_error("No depth sensor found");
}

void capture_loop(int width, int height, int fps) {
    while (g_running.load()) {
        try {
            rs2::pipeline pipe;
            rs2::config cfg;

            cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
            cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

            rs2::pipeline_profile profile = pipe.start(cfg);
            rs2::device dev = profile.get_device();

            {
                std::lock_guard<std::mutex> lock(g_meta_mutex);
                g_depth_scale = get_depth_scale_from_profile(profile);
                auto color_stream = profile.get_stream(RS2_STREAM_COLOR)
                                        .as<rs2::video_stream_profile>();
                g_color_intrinsics = color_stream.get_intrinsics();
                g_intrinsics_ready = true;
            }

            g_camera_connected.store(true);

            std::cout << "Connected to: "
                      << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            std::cout << "Depth scale: " << g_depth_scale << " meters/unit" << std::endl;
            std::cout << "RGB intrinsics: fx=" << g_color_intrinsics.fx
                      << " fy=" << g_color_intrinsics.fy
                      << " cx=" << g_color_intrinsics.ppx
                      << " cy=" << g_color_intrinsics.ppy << std::endl;

            rs2::align align_to_color(RS2_STREAM_COLOR);
            rs2::colorizer color_map;

            while (g_running.load()) {
                rs2::frameset frames = pipe.wait_for_frames();
                frames = align_to_color.process(frames);

                rs2::video_frame color_frame = frames.get_color_frame();
                rs2::depth_frame depth_frame = frames.get_depth_frame();
                rs2::frame depth_colorized = color_map.colorize(depth_frame);

                if (!color_frame || !depth_frame || !depth_colorized) {
                    continue;
                }

                cv::Mat rgb(
                    cv::Size(color_frame.get_width(), color_frame.get_height()),
                    CV_8UC3,
                    (void*)color_frame.get_data(),
                    cv::Mat::AUTO_STEP
                );

                cv::Mat depth_raw(
                    cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                    CV_16UC1,
                    (void*)depth_frame.get_data(),
                    cv::Mat::AUTO_STEP
                );

                rs2::video_frame depth_vis_vf = depth_colorized.as<rs2::video_frame>();
                cv::Mat depth_vis(
                    cv::Size(depth_vis_vf.get_width(), depth_vis_vf.get_height()),
                    CV_8UC3,
                    (void*)depth_vis_vf.get_data(),
                    cv::Mat::AUTO_STEP
                );

                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_frames.rgb_bgr = rgb.clone();
                    g_frames.depth_raw_16u = depth_raw.clone();
                    g_frames.depth_color_bgr = depth_vis.clone();
                    g_frames.timestamp = timestamp_now();
                    g_frames.frame_counter++;
                }

                g_cv.notify_all();
            }

            pipe.stop();
            g_camera_connected.store(false);

        } catch (const rs2::error& e) {
            g_camera_connected.store(false);
            std::cerr << "RealSense error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            g_camera_connected.store(false);
            std::cerr << "Capture loop exception: " << e.what() << std::endl;
        }

        if (g_running.load()) {
            std::cerr << "Retrying camera connection in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
}

bool get_latest_frames(cv::Mat& rgb, cv::Mat& depth_raw, cv::Mat& depth_color,
                       std::string& ts, uint64_t& counter) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_frames.rgb_bgr.empty() ||
        g_frames.depth_raw_16u.empty() ||
        g_frames.depth_color_bgr.empty()) {
        return false;
    }

    rgb = g_frames.rgb_bgr.clone();
    depth_raw = g_frames.depth_raw_16u.clone();
    depth_color = g_frames.depth_color_bgr.clone();
    ts = g_frames.timestamp;
    counter = g_frames.frame_counter;
    return true;
}

void mjpeg_stream_handler(httplib::Response& res, bool stream_rgb) {
    res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    res.set_header("Pragma", "no-cache");

    res.set_content_provider(
        "multipart/x-mixed-replace; boundary=frame",
        [stream_rgb](size_t, httplib::DataSink& sink) {
            uint64_t last_seen = 0;

            while (g_running.load()) {
                cv::Mat frame;

                {
                    std::unique_lock<std::mutex> lock(g_mutex);
                    g_cv.wait_for(lock, std::chrono::milliseconds(500), [&]() {
                        return !g_running.load() || g_frames.frame_counter > last_seen;
                    });

                    if (!g_running.load()) break;
                    if (g_frames.frame_counter == last_seen) continue;

                    frame = stream_rgb ? g_frames.rgb_bgr.clone()
                                       : g_frames.depth_color_bgr.clone();
                    last_seen = g_frames.frame_counter;
                }

                if (frame.empty()) continue;

                auto jpg = encode_jpeg(frame, 75);

                std::ostringstream header;
                header << "--frame\r\n";
                header << "Content-Type: image/jpeg\r\n";
                header << "Content-Length: " << jpg.size() << "\r\n\r\n";

                std::string header_str = header.str();

                if (!sink.write(header_str.c_str(), header_str.size())) break;
                if (!sink.write(reinterpret_cast<const char*>(jpg.data()), jpg.size())) break;
                if (!sink.write("\r\n", 2)) break;
            }

            sink.done();
            return true;
        }
    );
}

int main(int argc, char** argv) {
    std::string host = "0.0.0.0";
    int port = 8000;
    int width = 640;
    int height = 480;
    int fps = 15;

    if (argc >= 2) width = std::stoi(argv[1]);
    if (argc >= 3) height = std::stoi(argv[2]);
    if (argc >= 4) fps = std::stoi(argv[3]);

    std::thread cap_thread(capture_loop, width, height, fps);

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        cv::Mat rgb, depth_raw, depth_color;
        std::string ts;
        uint64_t counter = 0;
        bool frames_ok = get_latest_frames(rgb, depth_raw, depth_color, ts, counter);

        float depth_scale = 0.0f;
        rs2_intrinsics intr{};
        bool intr_ok = false;

        {
            std::lock_guard<std::mutex> lock(g_meta_mutex);
            depth_scale = g_depth_scale;
            intr = g_color_intrinsics;
            intr_ok = g_intrinsics_ready;
        }

        std::ostringstream oss;
        oss << "{"
            << "\"ok\":true,"
            << "\"camera_connected\":" << (g_camera_connected.load() ? "true" : "false") << ","
            << "\"frames_ready\":" << (frames_ok ? "true" : "false") << ","
            << "\"frame_counter\":" << counter << ","
            << "\"depth_scale\":" << depth_scale << ","
            << "\"intrinsics_ready\":" << (intr_ok ? "true" : "false") << ","
            << "\"fx\":" << intr.fx << ","
            << "\"fy\":" << intr.fy << ","
            << "\"cx\":" << intr.ppx << ","
            << "\"cy\":" << intr.ppy
            << "}";

        res.set_content(oss.str(), "application/json");
    });

    svr.Get("/stream/rgb.mjpg", [](const httplib::Request&, httplib::Response& res) {
        mjpeg_stream_handler(res, true);
    });

    svr.Get("/stream/depth.mjpg", [](const httplib::Request&, httplib::Response& res) {
        mjpeg_stream_handler(res, false);
    });

    svr.Post("/capture", [](const httplib::Request& req, httplib::Response& res) {
    try {
        if (!req.body.empty()) {
            auto body = nlohmann::json::parse(req.body);
            if (body.contains("trigger_at_ms") && body["trigger_at_ms"].is_number_unsigned()) {
                uint64_t trigger_at_ms = body["trigger_at_ms"].get<uint64_t>();
                wait_until_trigger_ms(trigger_at_ms);
            }
        }
    } catch (...) {
        // ignore bad JSON and continue immediate capture
    }

    cv::Mat rgb, depth_raw, depth_color;
    std::string ts;
    uint64_t counter = 0;

    if (!get_latest_frames(rgb, depth_raw, depth_color, ts, counter)) {
        res.status = 503;
        res.set_content("{\"ok\":false,\"error\":\"frames not ready\"}", "application/json");
        return;
    }

    float depth_scale = 0.0f;
    rs2_intrinsics intr{};
    bool intr_ok = false;

    {
        std::lock_guard<std::mutex> lock(g_meta_mutex);
        depth_scale = g_depth_scale;
        intr = g_color_intrinsics;
        intr_ok = g_intrinsics_ready;
    }

    if (!intr_ok) {
        res.status = 503;
        res.set_content("{\"ok\":false,\"error\":\"intrinsics not ready\"}", "application/json");
        return;
    }

    std::vector<uchar> rgb_jpg = encode_jpeg(rgb, 95);
    std::vector<uchar> depth_png;
    cv::imencode(".png", depth_raw, depth_png);

    res.set_header("X-Timestamp", ts);
    res.set_header("X-Frame-Counter", std::to_string(counter));
    res.set_header("X-RGB-Size", std::to_string(rgb_jpg.size()));
    res.set_header("X-Depth-Scale", std::to_string(depth_scale));
    res.set_header("X-FX", std::to_string(intr.fx));
    res.set_header("X-FY", std::to_string(intr.fy));
    res.set_header("X-CX", std::to_string(intr.ppx));
    res.set_header("X-CY", std::to_string(intr.ppy));
    res.set_header("X-Width", std::to_string(rgb.cols));
    res.set_header("X-Height", std::to_string(rgb.rows));
    res.set_header("Content-Type", "application/octet-stream");

    std::string body;
    body.reserve(rgb_jpg.size() + depth_png.size());
    body.append(reinterpret_cast<const char*>(rgb_jpg.data()), rgb_jpg.size());
    body.append(reinterpret_cast<const char*>(depth_png.data()), depth_png.size());

    res.body = std::move(body);
});

    std::cout << "Server listening on http://" << host << ":" << port << std::endl;
    std::cout << "RGB stream:   /stream/rgb.mjpg" << std::endl;
    std::cout << "Depth stream: /stream/depth.mjpg" << std::endl;

    svr.listen(host.c_str(), port);

    g_running.store(false);
    g_cv.notify_all();
    if (cap_thread.joinable()) cap_thread.join();

    return 0;
}