#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>
#include <algorithm>
#include <numeric>

// Must match rl_agent.h
const int INPUT_SIZE = 27;
const int HIDDEN_SIZE = 64;
const int OUTPUT_SIZE = 1;

static const char* INPUT_LABELS[INPUT_SIZE] = {
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
    "HMax", "Holes", "Bump",
    "P0", "P1", "P2", "P3", "P4", "P5", "P6",
    "Well", "HMean", "HRange",
    "R0", "R1", "R2", "R3"
};

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

struct Connection {
    int from_layer;
    int from_node;
    int to_layer;
    int to_node;
    double weight;
    bool visible;
    bool highlighted;

    Connection(int fl, int fn, int tl, int tn, double w)
        : from_layer(fl), from_node(fn), to_layer(tl), to_node(tn),
          weight(w), visible(true), highlighted(false) {}
};

// Minimal 5x7 bitmap font (A-Z, 0-9, space, punctuation)
static const unsigned char FONT5X7[][7] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space
    {0x04,0x04,0x04,0x04,0x04,0x00,0x04}, // !
    {0x0A,0x0A,0x00,0x00,0x00,0x00,0x00}, // "
    {0x0A,0x1F,0x0A,0x1F,0x0A,0x00,0x00}, // #
    {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04}, // $
    {0x19,0x19,0x02,0x04,0x08,0x13,0x13}, // %
    {0x08,0x14,0x14,0x08,0x15,0x12,0x0D}, // &
    {0x04,0x04,0x00,0x00,0x00,0x00,0x00}, // '
    {0x02,0x04,0x08,0x08,0x08,0x04,0x02}, // (
    {0x08,0x04,0x02,0x02,0x02,0x04,0x08}, // )
    {0x00,0x0A,0x04,0x1F,0x04,0x0A,0x00}, // *
    {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, // +
    {0x00,0x00,0x00,0x00,0x04,0x04,0x08}, // ,
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, // -
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, // .
    {0x01,0x02,0x04,0x08,0x10,0x00,0x00}, // /
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
    {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}, // 2
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, // 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
    {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}, // :
    {0x00,0x0C,0x0C,0x00,0x0C,0x04,0x08}, // ;
    {0x02,0x04,0x08,0x10,0x08,0x04,0x02}, // <
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // =
    {0x08,0x04,0x02,0x01,0x02,0x04,0x08}, // >
    {0x0E,0x11,0x01,0x02,0x04,0x00,0x04}, // ?
    {0x0E,0x11,0x17,0x15,0x17,0x10,0x0E}, // @
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // A
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C
    {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, // D
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // E
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // F
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, // G
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, // H
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, // I
    {0x01,0x01,0x01,0x01,0x11,0x11,0x0E}, // J
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // K
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, // L
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, // M
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // N
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // O
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, // P
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, // Q
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // R
    {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E}, // S
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // T
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, // U
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, // V
    {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}, // W
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, // X
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, // Y
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // Z
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // [
};

static int fontIndex(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    if (c == ' ') return 0;
    if (c >= '!' && c <= 'Z') return c - ' ';
    if (c == '[') return 59;
    return 0;
}

class WeightVisualizer {
private:
    SDL_Window* window;
    SDL_GLContext gl_context;
    std::string model_file;

    std::vector<std::vector<double>> weights1;
    std::vector<double> bias1;
    std::vector<std::vector<double>> weights2;
    std::vector<double> bias2;
    std::vector<Connection> connections;

    time_t last_file_mtime;
    bool file_exists;
    int window_width;
    int window_height;

    // View mode: 0=3D network, 1=2D heatmaps, 2=feature importance bars
    int view_mode;

    // 3D camera
    float camera_angle_x;
    float camera_angle_y;
    float camera_distance;
    float camera_pan_y;
    float camera_pan_z;
    int mouse_x, mouse_y;
    int click_x, click_y;
    bool mouse_dragging;
    bool pan_dragging;
    bool auto_rotate;

    // Filtering / interaction
    float weight_percentile;   // show top N% by |weight|
    int show_layer_links;      // 0=all, 1=input-hidden, 2=hidden-output
    int selected_layer;        // -1 = none
    int selected_node;

    std::string last_update_time;
    int update_count;
    double global_min_w;
    double global_max_w;
    double global_max_abs;

    GLUquadric* sphere_quad;

    void weightToRGB(double weight, double min_val, double max_val, float& r, float& g, float& b) const {
        double span = max_val - min_val;
        double normalized = (span > 1e-12) ? (weight - min_val) / span : 0.5;
        normalized = std::max(0.0, std::min(1.0, normalized));
        if (normalized < 0.5) {
            double t = normalized * 2.0;
            r = 1.0f;
            g = (float)t;
            b = (float)t;
        } else {
            double t = (normalized - 0.5) * 2.0;
            r = (float)(1.0 - t);
            g = (float)(1.0 - t);
            b = 1.0f;
        }
    }

    void signedWeightRGB(double weight, float& r, float& g, float& b, float& a) const {
        double mag = std::abs(weight) / std::max(1e-9, global_max_abs);
        mag = std::max(0.0, std::min(1.0, mag));
        if (weight >= 0) {
            r = 0.25f + 0.35f * (1.0f - (float)mag);
            g = 0.45f + 0.55f * (float)mag;
            b = 1.0f;
        } else {
            r = 1.0f;
            g = 0.25f + 0.35f * (1.0f - (float)mag);
            b = 0.25f + 0.35f * (1.0f - (float)mag);
        }
        a = 0.15f + 0.85f * (float)mag;
    }

    time_t getFileModificationTime(const std::string& filename) {
        struct stat file_stat;
        if (stat(filename.c_str(), &file_stat) == 0) return file_stat.st_mtime;
        return 0;
    }

    bool loadModel() {
        std::ifstream file(model_file);
        if (!file.is_open()) {
            file_exists = false;
            return false;
        }

        file_exists = true;
        std::string line;
        std::vector<std::string> data_lines;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            data_lines.push_back(line);
        }
        if ((int)data_lines.size() < INPUT_SIZE + 1 + HIDDEN_SIZE + 1) return false;

        weights1.assign(INPUT_SIZE, std::vector<double>());
        for (int i = 0; i < INPUT_SIZE; i++) {
            std::istringstream iss(data_lines[i]);
            double val;
            while (iss >> val) weights1[i].push_back(val);
            if ((int)weights1[i].size() != HIDDEN_SIZE) return false;
        }

        {
            std::istringstream iss(data_lines[INPUT_SIZE]);
            bias1.clear();
            double val;
            while (iss >> val) bias1.push_back(val);
            if ((int)bias1.size() != HIDDEN_SIZE) return false;
        }

        weights2.assign(HIDDEN_SIZE, std::vector<double>());
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            std::istringstream iss(data_lines[INPUT_SIZE + 1 + i]);
            double val;
            while (iss >> val) weights2[i].push_back(val);
            if ((int)weights2[i].size() != OUTPUT_SIZE) return false;
        }

        {
            std::istringstream iss(data_lines[INPUT_SIZE + 1 + HIDDEN_SIZE]);
            bias2.clear();
            double val;
            while (iss >> val) bias2.push_back(val);
            if ((int)bias2.size() != OUTPUT_SIZE) return false;
        }

        buildConnections();
        updateVisibility();
        recomputeStats();

        auto now = std::time(nullptr);
        auto time_info = *std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S");
        last_update_time = oss.str();
        update_count++;
        return true;
    }

    void recomputeStats() {
        global_min_w = 0;
        global_max_w = 0;
        global_max_abs = 1e-9;
        bool first = true;
        for (const auto& c : connections) {
            if (first) {
                global_min_w = global_max_w = c.weight;
                first = false;
            }
            global_min_w = std::min(global_min_w, c.weight);
            global_max_w = std::max(global_max_w, c.weight);
            global_max_abs = std::max(global_max_abs, std::abs(c.weight));
        }
    }

    void buildConnections() {
        connections.clear();
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                connections.emplace_back(0, i, 1, j, weights1[i][j]);
            }
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                connections.emplace_back(1, i, 2, j, weights2[i][j]);
            }
        }
    }

    void updateVisibility() {
        std::vector<double> mags;
        mags.reserve(connections.size());
        for (const auto& c : connections) {
            bool layer_ok = (show_layer_links == 0) ||
                            (show_layer_links == 1 && c.from_layer == 0) ||
                            (show_layer_links == 2 && c.from_layer == 1);
            if (layer_ok) mags.push_back(std::abs(c.weight));
        }
        double threshold = 0.0;
        if (!mags.empty() && weight_percentile < 100.0f) {
            std::sort(mags.begin(), mags.end());
            float keep = weight_percentile / 100.0f;
            size_t idx = (size_t)((1.0f - keep) * (mags.size() - 1));
            threshold = mags[idx];
        }

        for (auto& c : connections) {
            bool layer_ok = (show_layer_links == 0) ||
                            (show_layer_links == 1 && c.from_layer == 0) ||
                            (show_layer_links == 2 && c.from_layer == 1);
            bool selected_ok = true;
            if (selected_layer >= 0) {
                selected_ok = (c.from_layer == selected_layer && c.from_node == selected_node) ||
                              (c.to_layer == selected_layer && c.to_node == selected_node);
            }
            c.visible = layer_ok && selected_ok &&
                        (weight_percentile >= 100.0f || std::abs(c.weight) >= threshold);
            c.highlighted = selected_layer >= 0 && selected_ok;
        }
    }

    void layerGrid(int layer, int& cols, int& rows) const {
        if (layer == 0) { cols = 6; rows = 5; }
        else if (layer == 1) { cols = 8; rows = 8; }
        else { cols = 1; rows = 1; }
    }

    float layerX(int layer) const {
        if (layer == 0) return -4.0f;
        if (layer == 1) return 0.0f;
        return 4.0f;
    }

    Vec3 nodePosition(int layer, int node_index) const {
        int cols, rows;
        layerGrid(layer, cols, rows);
        float spacing = 0.42f;
        float start_y = -(rows - 1) * spacing / 2.0f;
        float start_z = -(cols - 1) * spacing / 2.0f;
        int c = node_index % cols;
        int r = node_index / cols;
        return Vec3(layerX(layer), start_y + r * spacing, start_z + c * spacing);
    }

    int layerNodeCount(int layer) const {
        if (layer == 0) return INPUT_SIZE;
        if (layer == 1) return HIDDEN_SIZE;
        return OUTPUT_SIZE;
    }

    double nodeImportance(int layer, int node) const {
        double sum = 0.0;
        int count = 0;
        for (const auto& c : connections) {
            if ((c.from_layer == layer && c.from_node == node) ||
                (c.to_layer == layer && c.to_node == node)) {
                sum += std::abs(c.weight);
                count++;
            }
        }
        return count ? sum / count : 0.0;
    }

    void setupPerspective() {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (double)window_width / std::max(1, window_height), 0.1, 100.0);
        glMatrixMode(GL_MODELVIEW);
    }

    void setupOrtho() {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, window_width, window_height, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    void drawText2D(float x, float y, const std::string& text, float scale = 2.0f,
                    float r = 1, float g = 1, float b = 1) {
        glColor3f(r, g, b);
        glBegin(GL_QUADS);
        float cx = x;
        for (char ch : text) {
            const unsigned char* glyph = FONT5X7[fontIndex(ch)];
            for (int row = 0; row < 7; row++) {
                for (int col = 0; col < 5; col++) {
                    if (glyph[row] & (1 << (4 - col))) {
                        float px = cx + col * scale;
                        float py = y + row * scale;
                        glVertex2f(px, py);
                        glVertex2f(px + scale, py);
                        glVertex2f(px + scale, py + scale);
                        glVertex2f(px, py + scale);
                    }
                }
            }
            cx += 6 * scale;
        }
        glEnd();
    }

    void drawHUD() {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        setupOrtho();

        // Top bar
        glColor4f(0.05f, 0.06f, 0.09f, 0.85f);
        glBegin(GL_QUADS);
        glVertex2f(0, 0); glVertex2f(window_width, 0);
        glVertex2f(window_width, 52); glVertex2f(0, 52);
        glEnd();

        std::string mode_name = (view_mode == 0) ? "3D NETWORK" :
                                (view_mode == 1) ? "2D HEATMAPS" : "FEATURE BARS";
        drawText2D(12, 10, mode_name, 2.2f, 0.4f, 0.95f, 0.7f);
        drawText2D(220, 10, "FILE " + model_file, 1.6f, 0.75f, 0.75f, 0.8f);
        drawText2D(12, 32, "UPD #" + std::to_string(update_count) + "  " + last_update_time, 1.5f, 0.55f, 0.9f, 0.55f);

        // Legend bar
        int lx = window_width - 340;
        drawText2D(lx, 10, "NEG", 1.5f, 1, 0.4f, 0.4f);
        for (int i = 0; i < 160; i++) {
            float t = i / 159.0f;
            float w = -1.0 + 2.0 * t;
            float r, g, b, a;
            double old = global_max_abs;
            global_max_abs = 1.0;
            signedWeightRGB(w, r, g, b, a);
            global_max_abs = old;
            glColor3f(r, g, b);
            glBegin(GL_QUADS);
            glVertex2f(lx + 40 + i, 10);
            glVertex2f(lx + 41 + i, 10);
            glVertex2f(lx + 41 + i, 24);
            glVertex2f(lx + 40 + i, 24);
            glEnd();
        }
        drawText2D(lx + 210, 10, "POS", 1.5f, 0.4f, 0.8f, 1);

        // Bottom help
        glColor4f(0.05f, 0.06f, 0.09f, 0.9f);
        glBegin(GL_QUADS);
        glVertex2f(0, window_height - 70);
        glVertex2f(window_width, window_height - 70);
        glVertex2f(window_width, window_height);
        glVertex2f(0, window_height);
        glEnd();

        drawText2D(12, window_height - 58,
                   "K:MODE  SPACE:AUTO-ROT  [/]:FILTER  1/2/0:LINKS  CLICK:SELECT  R:RELOAD  H:HELP  ESC:QUIT",
                   1.5f, 0.8f, 0.85f, 0.9f);

        std::ostringstream info;
        info << std::fixed << std::setprecision(1)
             << "TOP " << weight_percentile << "%  LINKS ";
        if (show_layer_links == 0) info << "ALL";
        else if (show_layer_links == 1) info << "IN->HID";
        else info << "HID->OUT";
        info << "  W=[" << std::setprecision(3) << global_min_w << "," << global_max_w << "]";
        drawText2D(12, window_height - 34, info.str(), 1.5f, 0.55f, 0.9f, 0.75f);

        if (selected_layer >= 0) {
            std::ostringstream sel;
            sel << "SELECTED L" << selected_layer << " N" << selected_node;
            if (selected_layer == 0 && selected_node < INPUT_SIZE) {
                sel << " (" << INPUT_LABELS[selected_node] << ")";
            }
            sel << "  avg|w|=" << std::setprecision(4) << nodeImportance(selected_layer, selected_node);
            drawText2D(12, window_height - 16, sel.str(), 1.5f, 1.0f, 0.95f, 0.35f);
        } else {
            drawText2D(12, window_height - 16, "DRAG:ORBIT  RMB:PAN  WHEEL:ZOOM  C:CLEAR SELECT", 1.5f, 0.6f, 0.65f, 0.7f);
        }

        setupPerspective();
        glEnable(GL_DEPTH_TEST);
    }

    void applyCamera() {
        glLoadIdentity();
        glTranslatef(0.0f, camera_pan_y, -camera_distance);
        glRotatef(camera_angle_x, 1.0f, 0.0f, 0.0f);
        glRotatef(camera_angle_y, 0.0f, 1.0f, 0.0f);
        glTranslatef(0.0f, 0.0f, camera_pan_z);
    }

    void drawSphere(float radius) {
        if (!sphere_quad) sphere_quad = gluNewQuadric();
        gluSphere(sphere_quad, radius, 14, 14);
    }

    void drawLayerNodes(int layer) {
        int n = layerNodeCount(layer);
        for (int i = 0; i < n; i++) {
            Vec3 p = nodePosition(layer, i);
            glPushMatrix();
            glTranslatef(p.x, p.y, p.z);

            bool selected = (selected_layer == layer && selected_node == i);
            float imp = (float)nodeImportance(layer, i);
            float norm = (float)(imp / std::max(1e-9, global_max_abs));
            float radius = selected ? 0.16f : (0.08f + 0.07f * std::min(1.0f, norm));

            if (selected) {
                glColor3f(1.0f, 0.9f, 0.2f);
            } else if (layer == 0) {
                glColor3f(0.95f, 0.45f + 0.3f * norm, 0.45f);
            } else if (layer == 1) {
                glColor3f(0.4f + 0.3f * norm, 0.95f, 0.5f);
            } else {
                glColor3f(0.45f, 0.55f, 1.0f);
            }

            // Bias glow for hidden/output
            if (layer == 1 && i < (int)bias1.size()) {
                float bias_t = (float)std::tanh(bias1[i]);
                glColor3f(0.35f + 0.4f * norm, 0.75f + 0.2f * bias_t, 0.4f);
            }

            drawSphere(radius);
            glPopMatrix();
        }
    }

    void drawConnections3D() {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);

        // Weak lines first
        for (const auto& c : connections) {
            if (!c.visible || c.highlighted) continue;
            float r, g, b, a;
            signedWeightRGB(c.weight, r, g, b, a);
            float thickness = 1.0f + 3.0f * (float)(std::abs(c.weight) / global_max_abs);
            glLineWidth(thickness);
            glColor4f(r, g, b, a * 0.55f);
            Vec3 a3 = nodePosition(c.from_layer, c.from_node);
            Vec3 b3 = nodePosition(c.to_layer, c.to_node);
            glBegin(GL_LINES);
            glVertex3f(a3.x, a3.y, a3.z);
            glVertex3f(b3.x, b3.y, b3.z);
            glEnd();
        }

        // Highlighted connections on top
        for (const auto& c : connections) {
            if (!c.visible || !c.highlighted) continue;
            float r, g, b, a;
            signedWeightRGB(c.weight, r, g, b, a);
            glLineWidth(4.0f);
            glColor4f(1.0f, 0.95f, 0.2f, 0.95f);
            Vec3 a3 = nodePosition(c.from_layer, c.from_node);
            Vec3 b3 = nodePosition(c.to_layer, c.to_node);
            glBegin(GL_LINES);
            glVertex3f(a3.x, a3.y, a3.z);
            glVertex3f(b3.x, b3.y, b3.z);
            glEnd();
            // tinted weight color underlay
            glLineWidth(2.0f);
            glColor4f(r, g, b, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(a3.x, a3.y, a3.z);
            glVertex3f(b3.x, b3.y, b3.z);
            glEnd();
        }

        glDepthMask(GL_TRUE);
        glLineWidth(1.0f);
    }

    void drawAxes() {
        glDisable(GL_LIGHTING);
        glLineWidth(1.5f);
        glBegin(GL_LINES);
        glColor3f(0.5f, 0.2f, 0.2f); glVertex3f(-5, -2.2f, 0); glVertex3f(5, -2.2f, 0);
        glColor3f(0.25f, 0.25f, 0.35f); glVertex3f(-5, -2.2f, -2); glVertex3f(-5, -2.2f, 2);
        glEnd();
    }

    void drawLayerLabels3D() {
        // Labels drawn in HUD orthographic after projection would be nicer;
        // here we place small marker quads near each layer.
        glDisable(GL_LIGHTING);
        const char* names[3] = {"INPUT 27", "HIDDEN 64", "OUT 1"};
        float xs[3] = {-4.0f, 0.0f, 4.0f};
        for (int i = 0; i < 3; i++) {
            glPushMatrix();
            glTranslatef(xs[i], -2.0f, 0.0f);
            if (i == 0) glColor3f(1, 0.5f, 0.5f);
            else if (i == 1) glColor3f(0.5f, 1, 0.5f);
            else glColor3f(0.5f, 0.6f, 1);
            glBegin(GL_QUADS);
            glVertex3f(-0.35f, 0, -0.08f);
            glVertex3f( 0.35f, 0, -0.08f);
            glVertex3f( 0.35f, 0,  0.08f);
            glVertex3f(-0.35f, 0,  0.08f);
            glEnd();
            glPopMatrix();
            (void)names[i];
        }
    }

    bool projectToScreen(const Vec3& world, float& sx, float& sy, float& depth) {
        GLdouble model[16], proj[16];
        GLint viewport[4];
        glGetDoublev(GL_MODELVIEW_MATRIX, model);
        glGetDoublev(GL_PROJECTION_MATRIX, proj);
        glGetIntegerv(GL_VIEWPORT, viewport);
        GLdouble wx, wy, wz;
        if (!gluProject(world.x, world.y, world.z, model, proj, viewport, &wx, &wy, &wz)) return false;
        sx = (float)wx;
        sy = (float)(viewport[3] - wy);
        depth = (float)wz;
        return depth >= 0.0f && depth <= 1.0f;
    }

    void pickNodeAt(int mx, int my) {
        setupPerspective();
        applyCamera();
        int best_layer = -1;
        int best_node = -1;
        float best_dist = 28.0f;
        float best_depth = 2.0f;

        for (int layer = 0; layer < 3; layer++) {
            int n = layerNodeCount(layer);
            for (int i = 0; i < n; i++) {
                float sx, sy, depth;
                if (!projectToScreen(nodePosition(layer, i), sx, sy, depth)) continue;
                float dx = sx - mx;
                float dy = sy - my;
                float d = std::sqrt(dx * dx + dy * dy);
                if (d < best_dist || (std::abs(d - best_dist) < 1.0f && depth < best_depth)) {
                    best_dist = d;
                    best_depth = depth;
                    best_layer = layer;
                    best_node = i;
                }
            }
        }

        if (best_layer >= 0) {
            selected_layer = best_layer;
            selected_node = best_node;
            updateVisibility();
            std::cout << "[SELECT] Layer " << selected_layer << " Node " << selected_node;
            if (selected_layer == 0) std::cout << " (" << INPUT_LABELS[selected_node] << ")";
            std::cout << "  avg|w|=" << std::fixed << std::setprecision(5)
                      << nodeImportance(selected_layer, selected_node) << std::endl;

            // Dump strongest connections for this node
            std::vector<const Connection*> related;
            for (const auto& c : connections) {
                if ((c.from_layer == selected_layer && c.from_node == selected_node) ||
                    (c.to_layer == selected_layer && c.to_node == selected_node)) {
                    related.push_back(&c);
                }
            }
            std::sort(related.begin(), related.end(),
                      [](const Connection* a, const Connection* b) {
                          return std::abs(a->weight) > std::abs(b->weight);
                      });
            int shown = std::min(8, (int)related.size());
            for (int i = 0; i < shown; i++) {
                const Connection* c = related[i];
                std::cout << "  L" << c->from_layer << "N" << c->from_node
                          << " -> L" << c->to_layer << "N" << c->to_node
                          << "  w=" << c->weight << std::endl;
            }
        }
    }

    void render3D() {
        glViewport(0, 0, window_width, window_height);
        glClearColor(0.07f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        setupPerspective();
        applyCamera();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (!file_exists || connections.empty()) {
            drawHUD();
            setupOrtho();
            glDisable(GL_DEPTH_TEST);
            drawText2D(window_width / 2.0f - 120, window_height / 2.0f, "NO MODEL LOADED", 2.5f, 1, 0.4f, 0.4f);
            SDL_GL_SwapWindow(window);
            return;
        }

        drawAxes();
        drawConnections3D();

        // Simple lighting for spheres
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        GLfloat light_pos[] = {2.0f, 5.0f, 4.0f, 1.0f};
        GLfloat ambient[] = {0.25f, 0.25f, 0.3f, 1.0f};
        GLfloat diffuse[] = {0.9f, 0.9f, 0.95f, 1.0f};
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);

        drawLayerNodes(0);
        drawLayerNodes(1);
        drawLayerNodes(2);
        glDisable(GL_LIGHTING);

        drawLayerLabels3D();
        drawHUD();

        // Overlay layer names
        setupOrtho();
        glDisable(GL_DEPTH_TEST);
        drawText2D(18, 64, "INPUT", 1.8f, 1.0f, 0.55f, 0.55f);
        drawText2D(window_width / 2 - 40, 64, "HIDDEN", 1.8f, 0.55f, 1.0f, 0.55f);
        drawText2D(window_width - 90, 64, "OUTPUT", 1.8f, 0.55f, 0.65f, 1.0f);
        setupPerspective();

        SDL_GL_SwapWindow(window);
    }

    void drawHeatmapGL(const std::vector<std::vector<double>>& mat,
                       float x, float y, float w, float h,
                       const std::string& title) {
        if (mat.empty() || mat[0].empty()) return;
        int rows = (int)mat.size();
        int cols = (int)mat[0].size();
        double min_v = mat[0][0], max_v = mat[0][0];
        for (const auto& row : mat) {
            for (double v : row) {
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
        }

        drawText2D(x, y - 18, title, 1.6f, 0.85f, 0.9f, 1.0f);

        float cell_w = w / cols;
        float cell_h = h / rows;
        glBegin(GL_QUADS);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float r, g, b;
                weightToRGB(mat[i][j], min_v, max_v, r, g, b);
                glColor3f(r, g, b);
                float px = x + j * cell_w;
                float py = y + i * cell_h;
                glVertex2f(px, py);
                glVertex2f(px + cell_w - 0.5f, py);
                glVertex2f(px + cell_w - 0.5f, py + cell_h - 0.5f);
                glVertex2f(px, py + cell_h - 0.5f);
            }
        }
        glEnd();
    }

    void drawBiasBarGL(const std::vector<double>& bias, float x, float y, float w, float h,
                       const std::string& title) {
        if (bias.empty()) return;
        double min_v = *std::min_element(bias.begin(), bias.end());
        double max_v = *std::max_element(bias.begin(), bias.end());
        drawText2D(x, y - 18, title, 1.6f, 0.85f, 0.9f, 1.0f);
        float cell = w / bias.size();
        glBegin(GL_QUADS);
        for (size_t i = 0; i < bias.size(); i++) {
            float r, g, b;
            weightToRGB(bias[i], min_v, max_v, r, g, b);
            glColor3f(r, g, b);
            float px = x + i * cell;
            glVertex2f(px, y);
            glVertex2f(px + cell - 0.5f, y);
            glVertex2f(px + cell - 0.5f, y + h);
            glVertex2f(px, y + h);
        }
        glEnd();
    }

    void render2DHeatmaps() {
        glViewport(0, 0, window_width, window_height);
        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        setupOrtho();

        if (!file_exists || weights1.empty()) {
            drawText2D(window_width / 2 - 120, window_height / 2, "NO MODEL LOADED", 2.5f, 1, 0.4f, 0.4f);
            drawHUD();
            SDL_GL_SwapWindow(window);
            return;
        }

        float margin = 40;
        float panel_w = (window_width - 3 * margin) / 2.0f;
        float panel_h = (window_height - 180) / 2.0f;

        drawHeatmapGL(weights1, margin, 80, panel_w, panel_h, "WEIGHTS1 INPUT x HIDDEN");
        drawBiasBarGL(bias1, margin * 2 + panel_w, 80, panel_w, panel_h * 0.35f, "BIAS1");
        drawHeatmapGL(weights2, margin, 100 + panel_h, panel_w * 0.35f, panel_h, "WEIGHTS2 HIDDEN x OUT");
        drawBiasBarGL(bias2, margin * 2 + panel_w, 100 + panel_h, panel_w * 0.2f, 40, "BIAS2");

        // Feature labels for first columns of attention
        for (int i = 0; i < INPUT_SIZE; i++) {
            float py = 80 + i * (panel_h / INPUT_SIZE);
            if (i % 2 == 0) drawText2D(4, py, INPUT_LABELS[i], 1.1f, 0.7f, 0.7f, 0.75f);
        }

        drawHUD();
        SDL_GL_SwapWindow(window);
    }

    void renderFeatureBars() {
        glViewport(0, 0, window_width, window_height);
        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        setupOrtho();

        if (!file_exists || weights1.empty()) {
            drawText2D(window_width / 2 - 120, window_height / 2, "NO MODEL LOADED", 2.5f, 1, 0.4f, 0.4f);
            drawHUD();
            SDL_GL_SwapWindow(window);
            return;
        }

        drawText2D(40, 70, "MEAN ABS WEIGHT PER INPUT FEATURE (IMPORTANCE)", 2.0f, 0.9f, 0.95f, 1.0f);

        std::vector<double> importance(INPUT_SIZE, 0.0);
        double max_imp = 1e-9;
        for (int i = 0; i < INPUT_SIZE; i++) {
            double s = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) s += std::abs(weights1[i][j]);
            importance[i] = s / HIDDEN_SIZE;
            max_imp = std::max(max_imp, importance[i]);
        }

        float left = 90;
        float top = 110;
        float bar_h = std::max(14.0f, (window_height - 220.0f) / INPUT_SIZE);
        float max_w = window_width - left - 80;

        for (int i = 0; i < INPUT_SIZE; i++) {
            float y = top + i * bar_h;
            float w = (float)(importance[i] / max_imp) * max_w;
            float r, g, b, a;
            signedWeightRGB(importance[i], r, g, b, a);
            // force positive blue-green scale
            r = 0.2f; g = 0.55f + 0.4f * (float)(importance[i] / max_imp); b = 0.95f;
            glColor3f(r, g, b);
            glBegin(GL_QUADS);
            glVertex2f(left, y);
            glVertex2f(left + w, y);
            glVertex2f(left + w, y + bar_h - 2);
            glVertex2f(left, y + bar_h - 2);
            glEnd();
            drawText2D(8, y + 2, INPUT_LABELS[i], 1.4f, 0.85f, 0.85f, 0.9f);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << importance[i];
            drawText2D(left + w + 8, y + 2, oss.str(), 1.3f, 0.7f, 0.8f, 0.85f);
        }

        // Hidden->output importance
        drawText2D(window_width / 2 + 20, 70, "HIDDEN->OUT |W|", 1.8f, 0.9f, 0.95f, 1.0f);
        double max_h = 1e-9;
        for (int i = 0; i < HIDDEN_SIZE; i++) max_h = std::max(max_h, std::abs(weights2[i][0]));
        float hx = window_width / 2 + 20;
        float hy = 110;
        float cell = std::min(18.0f, (window_height - 220.0f) / 8.0f);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            int row = i / 8;
            int col = i % 8;
            float mag = (float)(std::abs(weights2[i][0]) / max_h);
            float r, g, b, a;
            signedWeightRGB(weights2[i][0], r, g, b, a);
            glColor3f(r, g, b);
            float px = hx + col * (cell + 2);
            float py = hy + row * (cell + 2);
            glBegin(GL_QUADS);
            glVertex2f(px, py);
            glVertex2f(px + cell * (0.35f + 0.65f * mag), py);
            glVertex2f(px + cell * (0.35f + 0.65f * mag), py + cell);
            glVertex2f(px, py + cell);
            glEnd();
        }

        drawHUD();
        SDL_GL_SwapWindow(window);
    }

    void printHelp() const {
        std::cout << "\n=== Weight Visualizer Controls ===\n"
                  << "  K / Tab     Cycle views: 3D network -> 2D heatmaps -> feature bars\n"
                  << "  Mouse drag  Orbit camera (3D)\n"
                  << "  Right-drag  Pan camera (3D)\n"
                  << "  Wheel       Zoom (3D)\n"
                  << "  Space       Toggle auto-rotate\n"
                  << "  [ / ]       Show fewer / more connections (percentile filter)\n"
                  << "  1 / 2 / 0   Show IN->HID / HID->OUT / ALL links\n"
                  << "  Click       Select neuron (highlight its connections)\n"
                  << "  C           Clear selection\n"
                  << "  R           Reload model\n"
                  << "  H           This help\n"
                  << "  Esc / Q     Quit\n"
                  << "==================================\n\n";
    }

public:
    WeightVisualizer(const std::string& filename = "tetris_model.txt")
        : window(nullptr), gl_context(nullptr), model_file(filename),
          last_file_mtime(0), file_exists(false),
          window_width(1600), window_height(900), view_mode(0),
          camera_angle_x(22.0f), camera_angle_y(35.0f), camera_distance(11.0f),
          camera_pan_y(0.0f), camera_pan_z(0.0f),
          mouse_x(0), mouse_y(0), click_x(0), click_y(0),
          mouse_dragging(false), pan_dragging(false),
          auto_rotate(true),
          weight_percentile(8.0f), show_layer_links(0),
          selected_layer(-1), selected_node(0),
          update_count(0), global_min_w(0), global_max_w(0), global_max_abs(1),
          sphere_quad(nullptr) {

        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return;
        }

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

        window = SDL_CreateWindow(
            "Neural Network Weight Visualizer — 3D",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            window_width, window_height,
            SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL
        );
        if (!window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return;
        }

        gl_context = SDL_GL_CreateContext(window);
        if (!gl_context) {
            std::cerr << "OpenGL context creation failed: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return;
        }
        SDL_GL_SetSwapInterval(1);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        glClearColor(0.07f, 0.08f, 0.12f, 1.0f);
        setupPerspective();

        sphere_quad = gluNewQuadric();
        last_file_mtime = getFileModificationTime(model_file);
        loadModel();
        printHelp();
        std::cout << "Starting in 3D view. Press K to cycle modes.\n";
    }

    ~WeightVisualizer() {
        if (sphere_quad) gluDeleteQuadric(sphere_quad);
        if (gl_context) SDL_GL_DeleteContext(gl_context);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void run() {
        if (!window || !gl_context) return;

        bool running = true;
        SDL_Event event;
        Uint32 last_ticks = SDL_GetTicks();

        while (running) {
            Uint32 now = SDL_GetTicks();
            float dt = (now - last_ticks) / 1000.0f;
            last_ticks = now;

            time_t current_mtime = getFileModificationTime(model_file);
            if (current_mtime != last_file_mtime && current_mtime > 0) {
                if (loadModel()) {
                    std::cout << "\n=== MODEL UPDATED #" << update_count
                              << " @ " << last_update_time << " ===\n";
                    last_file_mtime = current_mtime;
                }
            }

            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    running = false;
                } else if (event.type == SDL_WINDOWEVENT &&
                           event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    window_width = event.window.data1;
                    window_height = event.window.data2;
                    glViewport(0, 0, window_width, window_height);
                } else if (event.type == SDL_KEYDOWN) {
                    switch (event.key.keysym.sym) {
                        case SDLK_ESCAPE:
                        case SDLK_q:
                            running = false;
                            break;
                        case SDLK_r:
                            loadModel();
                            last_file_mtime = getFileModificationTime(model_file);
                            std::cout << "Reloaded model.\n";
                            break;
                        case SDLK_k:
                        case SDLK_TAB:
                            view_mode = (view_mode + 1) % 3;
                            auto_rotate = (view_mode == 0);
                            std::cout << "View: "
                                      << (view_mode == 0 ? "3D network" :
                                          view_mode == 1 ? "2D heatmaps" : "feature bars")
                                      << std::endl;
                            SDL_SetWindowTitle(window,
                                view_mode == 0 ? "Neural Network Weight Visualizer — 3D" :
                                view_mode == 1 ? "Neural Network Weight Visualizer — Heatmaps" :
                                                 "Neural Network Weight Visualizer — Features");
                            break;
                        case SDLK_SPACE:
                            auto_rotate = !auto_rotate;
                            break;
                        case SDLK_LEFTBRACKET:
                            weight_percentile = std::max(1.0f, weight_percentile - 1.0f);
                            updateVisibility();
                            std::cout << "Showing top " << weight_percentile << "% connections\n";
                            break;
                        case SDLK_RIGHTBRACKET:
                            weight_percentile = std::min(100.0f, weight_percentile + 1.0f);
                            updateVisibility();
                            std::cout << "Showing top " << weight_percentile << "% connections\n";
                            break;
                        case SDLK_1:
                            show_layer_links = 1;
                            updateVisibility();
                            break;
                        case SDLK_2:
                            show_layer_links = 2;
                            updateVisibility();
                            break;
                        case SDLK_0:
                            show_layer_links = 0;
                            updateVisibility();
                            break;
                        case SDLK_c:
                            selected_layer = -1;
                            updateVisibility();
                            break;
                        case SDLK_h:
                            printHelp();
                            break;
                        case SDLK_EQUALS:
                        case SDLK_PLUS:
                            camera_distance = std::max(4.0f, camera_distance - 0.5f);
                            break;
                        case SDLK_MINUS:
                            camera_distance = std::min(30.0f, camera_distance + 0.5f);
                            break;
                        default:
                            break;
                    }
                } else if (event.type == SDL_MOUSEMOTION) {
                    mouse_x = event.motion.x;
                    mouse_y = event.motion.y;
                    if (view_mode == 0 && mouse_dragging) {
                        camera_angle_y += event.motion.xrel * 0.4f;
                        camera_angle_x += event.motion.yrel * 0.4f;
                        camera_angle_x = std::max(-89.0f, std::min(89.0f, camera_angle_x));
                        auto_rotate = false;
                    } else if (view_mode == 0 && pan_dragging) {
                        camera_pan_y -= event.motion.yrel * 0.01f;
                        camera_pan_z += event.motion.xrel * 0.01f;
                        auto_rotate = false;
                    }
                } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        if (view_mode == 0) {
                            mouse_dragging = true;
                            click_x = event.button.x;
                            click_y = event.button.y;
                        }
                    } else if (event.button.button == SDL_BUTTON_RIGHT) {
                        pan_dragging = true;
                    }
                } else if (event.type == SDL_MOUSEBUTTONUP) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        if (view_mode == 0 && mouse_dragging) {
                            int dx = event.button.x - click_x;
                            int dy = event.button.y - click_y;
                            if (dx * dx + dy * dy < 25) {
                                pickNodeAt(event.button.x, event.button.y);
                            }
                        }
                        mouse_dragging = false;
                    } else if (event.button.button == SDL_BUTTON_RIGHT) {
                        pan_dragging = false;
                    }
                } else if (event.type == SDL_MOUSEWHEEL) {
                    if (event.wheel.y > 0) camera_distance = std::max(4.0f, camera_distance - 0.4f);
                    else if (event.wheel.y < 0) camera_distance = std::min(30.0f, camera_distance + 0.4f);
                }
            }

            if (view_mode == 0 && auto_rotate) {
                camera_angle_y += 12.0f * dt;
            }

            if (view_mode == 0) render3D();
            else if (view_mode == 1) render2DHeatmaps();
            else renderFeatureBars();
        }
    }
};

int main(int argc, char* argv[]) {
    std::string model_file = "tetris_model.txt";
    if (argc > 1) model_file = argv[1];

    WeightVisualizer visualizer(model_file);
    visualizer.run();
    return 0;
}
