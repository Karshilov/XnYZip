#ifndef XnYZip_TIMER_HPP
#define XnYZip_TIMER_HPP

#include <string>
#include <iostream>
#include <chrono>

namespace XnYZip {
    class Timer {
    public:
        Timer() = default;

        Timer(bool initstart) {
            if (initstart) {
                start();
            }
        }

        void start() {
            begin = std::chrono::steady_clock::now();
        }

        double stop() {
            end = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(end - begin).count();
        }

        double stop(const std::string &msg) {
            double seconds = stop();
            std::cout << msg << " time = " << seconds << "s" << std::endl;
            return seconds;
        }

    private:
        std::chrono::time_point<std::chrono::steady_clock> begin, end;
    };
};


#endif //XnYZip_TIMER_HPP
