#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct SharedFrames {
    cv::Mat color;
    cv::Mat depth_vis;
    rs2::depth_frame depth_raw;
    std::mutex mtx;
    std::atomic<bool> ready{false};
};

static bool send_all(int sock, const void* data, size_t len) {
    const char* ptr = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t sent = send(sock, ptr, len, 0);
        if (sent <= 0) return false;
        ptr += sent;
        len -= sent;
    }
    return true;
}

static int create_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    return server_fd;
}

static std::string http_ok(const std::string& content_type, size_t content_length) {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << content_length << "\r\n"
        << "Cache-Control: no-cache\r\n"
        << "Connection: close\r\n\r\n";
    return oss.str();
}

static std::string http_not_found() {
    const std::string body = "404 Not Found\n";
    std::ostringstream oss;
    oss << "HTTP/1.1 404 Not Found\r\n"
        << "Content-Type: text/plain\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

static std::string http_bad_request(const std::string& msg) {
    std::ostringstream body;
    body << "400 Bad Request\n" << msg << "\n";
    std::string s = body.str();

    std::ostringstream oss;
    oss << "HTTP/1.1 400 Bad Request\r\n"
        << "Content-Type: text/plain\r\n"
        << "Content-Length: " << s.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << s;
    return oss.str();
}

static std::string html_index() {
    const std::string body =
        "<html><head><title>D435 Preview</title></head>"
        "<body style='font-family:sans-serif'>"
        "<h2>D435 Preview Server</h2>"
        "<ul>"
        "<li><a href='/color'>/color</a></li>"
        "<li><a href='/depth'>/depth</a></li>"
        "<li><a href='/combined'>/combined</a></li>"
        "<li><a href='/depth_value?x=320&y=240'>/depth_value?x=320&y=240</a></li>"
        "</ul>"
        "<h3>Combined Preview</h3>"
        "<img src='/combined' style='max-width:100%; border:1px solid #ccc;'/>"
        "</body></html>";
    return body;
}

static bool encode_jpeg(const cv::Mat& img, std::vector<uchar>& out) {
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
    return cv::imencode(".jpg", img, out, params);
}

static void capture_loop(SharedFrames& shared) {
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(rs2::stream::color, 640, 480, rs2::format::bgr8, 15);
    cfg.enable_stream(rs2::stream::depth, 640, 480, rs2::format::z16, 15);

    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::align align_to_color(rs2::stream::color);
    rs2::colorizer colorizer;

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            continue;
        }

        rs2::frame depth_colorized = colorizer.process(depth_frame);
        rs2::video_frame depth_vf = depth_colorized.as<rs2::video_frame>();

        cv::Mat color(
            cv::Size(color_frame.get_width(), color_frame.get_height()),
            CV_8UC3,
            (void*)color_frame.get_data(),
            cv::Mat::AUTO_STEP
        );

        cv::Mat depth_vis(
            cv::Size(depth_vf.get_width(), depth_vf.get_height()),
            CV_8UC3,
            (void*)depth_vf.get_data(),
            cv::Mat::AUTO_STEP
        );

        {
            std::lock_guard<std::mutex> lock(shared.mtx);
            shared.color = color.clone();
            shared.depth_vis = depth_vis.clone();
            shared.depth_raw = depth_frame;
            shared.ready = true;
        }
    }
}

static std::string get_path(const std::string& req) {
    size_t method_end = req.find(' ');
    if (method_end == std::string::npos) return "/";
    size_t path_end = req.find(' ', method_end + 1);
    if (path_end == std::string::npos) return "/";
    return req.substr(method_end + 1, path_end - method_end - 1);
}

static bool parse_xy(const std::string& path, int& x, int& y) {
    size_t q = path.find('?');
    if (q == std::string::npos) return false;

    std::string query = path.substr(q + 1);
    std::istringstream iss(query);
    std::string token;
    bool has_x = false, has_y = false;

    while (std::getline(iss, token, '&')) {
        size_t eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = token.substr(0, eq);
        std::string value = token.substr(eq + 1);

        try {
            if (key == "x") {
                x = std::stoi(value);
                has_x = true;
            } else if (key == "y") {
                y = std::stoi(value);
                has_y = true;
            }
        } catch (...) {
            return false;
        }
    }

    return has_x && has_y;
}

static void handle_mjpeg_stream(int client_fd, const cv::Mat& frame) {
    std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

    if (!send_all(client_fd, header.c_str(), header.size())) {
        return;
    }

    while (true) {
        std::vector<uchar> jpg;
        if (!encode_jpeg(frame, jpg)) {
            break;
        }

        std::ostringstream part;
        part << "--frame\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpg.size() << "\r\n\r\n";
        std::string part_header = part.str();

        if (!send_all(client_fd, part_header.c_str(), part_header.size()) ||
            !send_all(client_fd, jpg.data(), jpg.size()) ||
            !send_all(client_fd, "\r\n", 2)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(66));
    }
}

static void client_thread(int client_fd, SharedFrames& shared) {
    char buffer[4096];
    ssize_t r = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (r <= 0) {
        close(client_fd);
        return;
    }
    buffer[r] = '\0';

    std::string req(buffer);
    std::string path = get_path(req);

    if (path == "/") {
        std::string body = html_index();
        std::string resp = http_ok("text/html", body.size()) + body;
        send_all(client_fd, resp.c_str(), resp.size());
        close(client_fd);
        return;
    }

    if (!shared.ready.load()) {
        std::string resp = http_bad_request("Frames not ready yet");
        send_all(client_fd, resp.c_str(), resp.size());
        close(client_fd);
        return;
    }

    if (path == "/color" || path == "/depth" || path == "/combined") {
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Cache-Control: no-cache\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

        if (!send_all(client_fd, header.c_str(), header.size())) {
            close(client_fd);
            return;
        }

        while (true) {
            cv::Mat img;
            {
                std::lock_guard<std::mutex> lock(shared.mtx);
                if (path == "/color") {
                    img = shared.color.clone();
                } else if (path == "/depth") {
                    img = shared.depth_vis.clone();
                } else {
                    cv::hconcat(shared.color, shared.depth_vis, img);
                }
            }

            if (img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            std::vector<uchar> jpg;
            if (!encode_jpeg(img, jpg)) {
                break;
            }

            std::ostringstream part;
            part << "--frame\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Content-Length: " << jpg.size() << "\r\n\r\n";
            std::string part_header = part.str();

            if (!send_all(client_fd, part_header.c_str(), part_header.size()) ||
                !send_all(client_fd, jpg.data(), jpg.size()) ||
                !send_all(client_fd, "\r\n", 2)) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(66));
        }

        close(client_fd);
        return;
    }

    if (path.rfind("/depth_value", 0) == 0) {
        int x = 0, y = 0;
        if (!parse_xy(path, x, y)) {
            std::string resp = http_bad_request("Use /depth_value?x=320&y=240");
            send_all(client_fd, resp.c_str(), resp.size());
            close(client_fd);
            return;
        }

        float depth_m = -1.0f;
        int width = 0, height = 0;

        {
            std::lock_guard<std::mutex> lock(shared.mtx);
            width = shared.depth_raw.get_width();
            height = shared.depth_raw.get_height();

            if (x < 0 || y < 0 || x >= width || y >= height) {
                std::ostringstream oss;
                oss << "x,y out of range. Valid range: x=[0," << (width - 1)
                    << "], y=[0," << (height - 1) << "]";
                std::string resp = http_bad_request(oss.str());
                send_all(client_fd, resp.c_str(), resp.size());
                close(client_fd);
                return;
            }

            depth_m = shared.depth_raw.get_distance(x, y);
        }

        std::ostringstream body;
        body << "{\n"
             << "  \"x\": " << x << ",\n"
             << "  \"y\": " << y << ",\n"
             << "  \"depth_m\": " << depth_m << "\n"
             << "}\n";

        std::string b = body.str();
        std::string resp = http_ok("application/json", b.size()) + b;
        send_all(client_fd, resp.c_str(), resp.size());
        close(client_fd);
        return;
    }

    std::string resp = http_not_found();
    send_all(client_fd, resp.c_str(), resp.size());
    close(client_fd);
}

int main() {
    try {
        SharedFrames shared;

        std::thread cap_thread(capture_loop, std::ref(shared));
        cap_thread.detach();

        int server_fd = create_server(8080);
        if (server_fd < 0) return 1;

        std::cout << "D435 preview server running on port 8080\n";
        std::cout << "Open on laptop:\n";
        std::cout << "  http://<PI4_IP>:8080/\n";
        std::cout << "  http://<PI4_IP>:8080/color\n";
        std::cout << "  http://<PI4_IP>:8080/depth\n";
        std::cout << "  http://<PI4_IP>:8080/combined\n";
        std::cout << "  http://<PI4_IP>:8080/depth_value?x=320&y=240\n";

        while (true) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                perror("accept");
                continue;
            }

            std::thread(client_thread, client_fd, std::ref(shared)).detach();
        }

        close(server_fd);
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}