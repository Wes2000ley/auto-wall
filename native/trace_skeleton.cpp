// Trace the same ordered pixel graph as wall_geometry.py. No image/model dependencies.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#ifdef _WIN32
#define AW_EXPORT extern "C" __declspec(dllexport)
#else
#define AW_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct TraceOutput {
    std::int32_t* points;
    std::int64_t* offsets;
    std::int64_t point_count;
    std::int64_t path_count;
};

namespace {
constexpr int dy[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
constexpr int dx[8] = {0, -1, 1, 0, -1, 1, -1, 1};
constexpr int reverse[8] = {3, 2, 1, 0, 7, 6, 5, 4};

int degree(std::uint8_t bits) {
    int count = 0;
    for (; bits; bits &= bits - 1) ++count;
    return count;
}
}

// Status: 0 success, 1 invalid input, 2 cancellation, 3 allocation/internal failure.
// points are flattened row-major pixel indices. offsets includes the final end.
AW_EXPORT int aw_trace_skeleton(const std::uint8_t* pixels, std::int32_t width,
                               std::int32_t height, int (*cancelled)(), TraceOutput* output) noexcept {
    if (!output) return 1;
    *output = {};
    const auto cells = static_cast<std::int64_t>(width) * height;
    if (!pixels || width <= 0 || height <= 0 || cells > std::numeric_limits<std::int32_t>::max()) return 1;
    try {
        std::vector<std::uint8_t> neighbors(static_cast<std::size_t>(cells), 0);
        std::vector<std::uint8_t> visited(static_cast<std::size_t>(cells), 0);
        for (std::int32_t index = 0; index < cells; ++index) {
            if ((index & 4095) == 0 && cancelled && cancelled()) return 2;
            if (!pixels[index]) continue;
            const int row = index / width, column = index % width;
            for (int direction = 0; direction < 8; ++direction) {
                const int y = row + dy[direction], x = column + dx[direction];
                if (y < 0 || y >= height || x < 0 || x >= width || !pixels[y * width + x]) continue;
                // Do not add triangular shortcuts across orthogonal junctions.
                if (dx[direction] && dy[direction] &&
                    (pixels[y * width + column] || pixels[row * width + x])) continue;
                neighbors[index] |= static_cast<std::uint8_t>(1 << direction);
            }
        }
        std::vector<std::int32_t> points;
        std::vector<std::int64_t> offsets{0};
        // Junctions/endpoints first, then closed loops. Both passes are row-major.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::int32_t start = 0; start < cells; ++start) {
                if ((start & 4095) == 0 && cancelled && cancelled()) return 2;
                if (!neighbors[start] || (degree(neighbors[start]) == 2) != (pass == 1)) continue;
                for (int direction = 0; direction < 8; ++direction) {
                    const auto bit = static_cast<std::uint8_t>(1 << direction);
                    if (!(neighbors[start] & bit) || (visited[start] & bit)) continue;
                    points.push_back(start);
                    auto previous = start;
                    auto current = start + dy[direction] * width + dx[direction];
                    int step = direction;
                    while (true) {
                        if ((points.size() & 4095) == 0 && cancelled && cancelled()) return 2;
                        visited[previous] |= static_cast<std::uint8_t>(1 << step);
                        visited[current] |= static_cast<std::uint8_t>(1 << reverse[step]);
                        points.push_back(current);
                        if (current == start || degree(neighbors[current]) != 2) break;
                        int following = 0;
                        for (; following < 8; ++following) {
                            if (!(neighbors[current] & (1 << following))) continue;
                            const auto candidate = current + dy[following] * width + dx[following];
                            if (candidate != previous) break;
                        }
                        if (following == 8 || (visited[current] & (1 << following))) break;
                        previous = current;
                        current += dy[following] * width + dx[following];
                        step = following;
                    }
                    offsets.push_back(static_cast<std::int64_t>(points.size()));
                }
            }
        }
        auto point_buffer = std::make_unique<std::int32_t[]>(points.size());
        auto offset_buffer = std::make_unique<std::int64_t[]>(offsets.size());
        std::copy(points.begin(), points.end(), point_buffer.get());
        std::copy(offsets.begin(), offsets.end(), offset_buffer.get());
        output->point_count = static_cast<std::int64_t>(points.size());
        output->path_count = static_cast<std::int64_t>(offsets.size()) - 1;
        output->points = point_buffer.release();
        output->offsets = offset_buffer.release();
        return 0;
    } catch (...) {
        return 3;  // Never propagate a C++ exception across the C ABI.
    }
}

AW_EXPORT void aw_free_trace(TraceOutput* output) noexcept {
    if (!output) return;
    delete[] output->points;
    delete[] output->offsets;
    *output = {};
}

AW_EXPORT int aw_geometry_abi() noexcept { return 1; }
