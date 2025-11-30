#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <GL/gl.h>
#include <GL/glu.h>
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

// Network architecture constants (matching rl_agent.h)
const int INPUT_SIZE = 29;
const int HIDDEN_SIZE = 64;
const int OUTPUT_SIZE = 1;

struct Connection {
    int from_layer;
    int from_node;
    int to_layer;
    int to_node;
    double weight;
    bool hovered;
    
    Connection(int fl, int fn, int tl, int tn, double w) 
        : from_layer(fl), from_node(fn), to_layer(tl), to_node(tn), weight(w), hovered(false) {}
};

class WeightVisualizer {
private:
    SDL_Window* window;
    SDL_GLContext gl_context;
    SDL_Renderer* renderer;  // For 2D mode
    std::string model_file;
    
    // Weight data
    std::vector<std::vector<double>> weights1;  // 29 x 64
    std::vector<double> bias1;                   // 64
    std::vector<std::vector<double>> weights2;   // 64 x 1
    std::vector<double> bias2;                   // 1
    
    // 3D connections
    std::vector<Connection> connections;
    
    // File monitoring
    time_t last_file_mtime;
    bool file_exists;
    
    // Display settings
    int window_width;
    int window_height;
    bool view_3d;
    
    // 3D camera
    float camera_angle_x;
    float camera_angle_y;
    float camera_distance;
    int mouse_x, mouse_y;
    bool mouse_dragging;
    
    // Timestamp
    std::string last_update_time;
    int update_count;
    
    // Hover info
    Connection* hovered_connection;
    std::string hover_text;
    
    SDL_Color weightToColor(double weight, double min_val, double max_val) {
        SDL_Color color;
        double normalized = (weight - min_val) / (max_val - min_val);
        normalized = std::max(0.0, std::min(1.0, normalized));
        
        if (normalized < 0.5) {
            double t = normalized * 2.0;
            color.r = 255;
            color.g = (Uint8)(255 * t);
            color.b = (Uint8)(255 * t);
        } else {
            double t = (normalized - 0.5) * 2.0;
            color.r = (Uint8)(255 * (1.0 - t));
            color.g = (Uint8)(255 * (1.0 - t));
            color.b = 255;
        }
        color.a = 255;
        return color;
    }
    
    void glColorFromWeight(double weight, double min_val, double max_val) {
        double normalized = (weight - min_val) / (max_val - min_val);
        normalized = std::max(0.0, std::min(1.0, normalized));
        
        if (normalized < 0.5) {
            double t = normalized * 2.0;
            glColor3f(1.0f, (float)t, (float)t);  // Red to White
        } else {
            double t = (normalized - 0.5) * 2.0;
            glColor3f((float)(1.0 - t), (float)(1.0 - t), 1.0f);  // White to Blue
        }
    }
    
    time_t getFileModificationTime(const std::string& filename) {
        struct stat file_stat;
        if (stat(filename.c_str(), &file_stat) == 0) {
            return file_stat.st_mtime;
        }
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
        
        if (data_lines.size() < INPUT_SIZE + 1 + HIDDEN_SIZE + 1) {
            return false;
        }
        
        // Load weights1
        weights1.clear();
        weights1.resize(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; i++) {
            std::istringstream iss(data_lines[i]);
            weights1[i].clear();
            double val;
            while (iss >> val) {
                weights1[i].push_back(val);
            }
            if (weights1[i].size() != HIDDEN_SIZE) return false;
        }
        
        // Load bias1
        std::istringstream iss1(data_lines[INPUT_SIZE]);
        bias1.clear();
        double val;
        while (iss1 >> val) {
            bias1.push_back(val);
        }
        if (bias1.size() != HIDDEN_SIZE) return false;
        
        // Load weights2
        weights2.clear();
        weights2.resize(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            std::istringstream iss(data_lines[INPUT_SIZE + 1 + i]);
            weights2[i].clear();
            while (iss >> val) {
                weights2[i].push_back(val);
            }
            if (weights2[i].size() != OUTPUT_SIZE) return false;
        }
        
        // Load bias2
        std::istringstream iss2(data_lines[INPUT_SIZE + 1 + HIDDEN_SIZE]);
        bias2.clear();
        while (iss2 >> val) {
            bias2.push_back(val);
        }
        if (bias2.size() != OUTPUT_SIZE) return false;
        
        // Build connections for 3D view
        buildConnections();
        
        // Update timestamp
        auto now = std::time(nullptr);
        auto time_info = *std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S");
        last_update_time = oss.str();
        update_count++;
        
        return true;
    }
    
    void buildConnections() {
        connections.clear();
        
        // Input to Hidden connections
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                connections.push_back(Connection(0, i, 1, j, weights1[i][j]));
            }
        }
        
        // Hidden to Output connections
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                connections.push_back(Connection(1, i, 2, j, weights2[i][j]));
            }
        }
    }
    
    void render2D() {
        // Make sure we're not using OpenGL context
        SDL_GL_MakeCurrent(window, nullptr);
        
        if (!renderer) {
            // Recreate renderer if needed
            renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
            if (!renderer) {
                std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
                return;
            }
        }
        
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        
        if (!file_exists || weights1.empty()) {
            SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
            SDL_Rect msg_rect = {window_width/2 - 200, window_height/2 - 30, 400, 60};
            SDL_RenderFillRect(renderer, &msg_rect);
            SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
            SDL_Rect border = {window_width/2 - 200, window_height/2 - 30, 400, 60};
            SDL_RenderDrawRect(renderer, &border);
            SDL_RenderPresent(renderer);
            return;
        }
        
        int margin = 20;
        int panel_width = (window_width - 3 * margin) / 2;
        int panel_height = (window_height - 120) / 2;
        
        // Draw matrices (simplified 2D view)
        drawWeightMatrix2D(weights1, margin, 50, panel_width, panel_height);
        drawBiasVector2D(bias1, margin + panel_width + margin, 50, panel_width, panel_height);
        drawWeightMatrix2D(weights2, margin, 50 + panel_height + margin, panel_width, panel_height);
        drawBiasVector2D(bias2, margin + panel_width + margin, 50 + panel_height + margin, panel_width, panel_height);
        
        // Draw info bar
        SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
        SDL_Rect info_rect = {0, window_height - 80, window_width, 80};
        SDL_RenderFillRect(renderer, &info_rect);
        
        // Draw mode indicator
        SDL_SetRenderDrawColor(renderer, 100, 200, 100, 255);
        SDL_Rect mode_rect = {margin, window_height - 70, 200, 30};
        SDL_RenderFillRect(renderer, &mode_rect);
        
        // Draw legend
        int legend_x = margin;
        int legend_y = window_height - 60;
        int legend_width = 300;
        int legend_height = 20;
        
        // Red to Blue gradient
        for (int i = 0; i < legend_width; i++) {
            double normalized = (double)i / legend_width;
            SDL_Color color = weightToColor(normalized * 2.0 - 1.0, -1.0, 1.0);
            SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
            SDL_RenderDrawLine(renderer, legend_x + i, legend_y, legend_x + i, legend_y + legend_height);
        }
        
        // Legend border
        SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
        SDL_Rect legend_border = {legend_x, legend_y, legend_width, legend_height};
        SDL_RenderDrawRect(renderer, &legend_border);
        
        // Timestamp indicator
        SDL_SetRenderDrawColor(renderer, 0, 200, 0, 255);
        SDL_Rect update_indicator = {legend_x + legend_width + 50, legend_y, 10, 10};
        SDL_RenderFillRect(renderer, &update_indicator);
        
        SDL_RenderPresent(renderer);
    }
    
    void drawWeightMatrix2D(const std::vector<std::vector<double>>& weights, 
                           int start_x, int start_y, int width, int height) {
        if (weights.empty() || weights[0].empty()) return;
        
        int rows = weights.size();
        int cols = weights[0].size();
        int cell_w = width / cols;
        int cell_h = height / rows;
        int cell_size = std::min(cell_w, cell_h);
        
        double min_val = weights[0][0];
        double max_val = weights[0][0];
        for (const auto& row : weights) {
            for (double w : row) {
                min_val = std::min(min_val, w);
                max_val = std::max(max_val, w);
            }
        }
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int x = start_x + j * cell_size;
                int y = start_y + i * cell_size;
                SDL_Color color = weightToColor(weights[i][j], min_val, max_val);
                SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
                SDL_Rect rect = {x, y, cell_size - 1, cell_size - 1};
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
    
    void drawBiasVector2D(const std::vector<double>& bias, int start_x, int start_y, int width, int height) {
        if (bias.empty()) return;
        int size = bias.size();
        int cell_w = width / size;
        int cell_size = std::min(cell_w, height);
        double min_val = *std::min_element(bias.begin(), bias.end());
        double max_val = *std::max_element(bias.begin(), bias.end());
        
        for (int i = 0; i < size; i++) {
            int x = start_x + i * cell_size;
            SDL_Color color = weightToColor(bias[i], min_val, max_val);
            SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
            SDL_Rect rect = {x, start_y, cell_size - 1, cell_size};
            SDL_RenderFillRect(renderer, &rect);
        }
    }
    
    void render3D() {
        // Make sure we're using OpenGL context
        SDL_GL_MakeCurrent(window, gl_context);
        
        glViewport(0, 0, window_width, window_height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        
        if (!file_exists || connections.empty()) {
            SDL_GL_SwapWindow(window);
            return;
        }
        
        // Setup camera
        glTranslatef(0.0f, 0.0f, -camera_distance);
        glRotatef(camera_angle_x, 1.0f, 0.0f, 0.0f);
        glRotatef(camera_angle_y, 0.0f, 1.0f, 0.0f);
        
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // Draw layers as nodes in 3D grid layout
        float layer_spacing = 4.0f;
        
        // Input layer (29 nodes) - arranged in 3D grid
        glPushMatrix();
        glTranslatef(-layer_spacing, 0.0f, 0.0f);
        drawLayer3DGrid(INPUT_SIZE, 0);
        glPopMatrix();
        
        // Hidden layer (64 nodes) - arranged in 3D grid (8x8)
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, 0.0f);
        drawLayer3DGrid(HIDDEN_SIZE, 1);
        glPopMatrix();
        
        // Output layer (1 node) - centered
        glPushMatrix();
        glTranslatef(layer_spacing, 0.0f, 0.0f);
        drawLayer3DGrid(OUTPUT_SIZE, 2);
        glPopMatrix();
        
        // Draw connections
        drawConnections3D();
        
        // Draw hover info
        if (hovered_connection) {
            drawHoverInfo();
        }
        
        SDL_GL_SwapWindow(window);
    }
    
    void drawLayer3DGrid(int node_count, int layer_id) {
        // Calculate grid dimensions for 3D layout
        int cols, rows;
        
        if (layer_id == 0) {
            // Input layer: 29 nodes -> 5x6 grid (30 positions, one empty)
            cols = 5;
            rows = 6;
        } else if (layer_id == 1) {
            // Hidden layer: 64 nodes -> 8x8 grid
            cols = 8;
            rows = 8;
        } else {
            // Output layer: 1 node -> centered
            cols = 1;
            rows = 1;
        }
        
        float node_spacing = 0.4f;
        // Grid is in y-z plane (x is layer position, set by parent glTranslatef)
        float start_y = -(rows - 1) * node_spacing / 2.0f;
        float start_z = -(cols - 1) * node_spacing / 2.0f;
        
        int node_idx = 0;
        for (int r = 0; r < rows && node_idx < node_count; r++) {
            for (int c = 0; c < cols && node_idx < node_count; c++) {
                glPushMatrix();
                // x is 0 (layer position set by parent), y is rows, z is columns
                float y = start_y + r * node_spacing;
                float z = start_z + c * node_spacing;
                glTranslatef(0.0f, y, z);
                
                // Node color based on layer
                if (layer_id == 0) glColor3f(1.0f, 0.5f, 0.5f);  // Light red
                else if (layer_id == 1) glColor3f(0.5f, 1.0f, 0.5f);  // Light green
                else glColor3f(0.5f, 0.5f, 1.0f);  // Light blue
                
                // Draw node as sphere
                GLUquadric* quad = gluNewQuadric();
                gluSphere(quad, 0.12f, 16, 16);
                gluDeleteQuadric(quad);
                glPopMatrix();
                
                node_idx++;
            }
        }
    }
    
    void getNodePosition3D(int layer, int node_index, float& y, float& z, float& x_offset) {
        // Returns y, z position within layer (x_offset will be set by layer spacing)
        int cols, rows;
        
        if (layer == 0) {
            // Input layer: 29 nodes -> 5x6 grid
            cols = 5;
            rows = 6;
        } else if (layer == 1) {
            // Hidden layer: 64 nodes -> 8x8 grid
            cols = 8;
            rows = 8;
        } else {
            // Output layer: 1 node -> centered
            cols = 1;
            rows = 1;
        }
        
        float node_spacing = 0.4f;
        float start_y = -(rows - 1) * node_spacing / 2.0f;
        float start_z = -(cols - 1) * node_spacing / 2.0f;
        
        int c = node_index % cols;
        int r = node_index / cols;
        
        y = start_y + r * node_spacing;
        z = start_z + c * node_spacing;
        x_offset = 0.0f;  // Will be set by layer spacing
    }
    
    void drawConnections3D() {
        if (connections.empty()) return;
        
        // Find min/max weight for color mapping
        double min_weight = connections[0].weight;
        double max_weight = connections[0].weight;
        for (const auto& conn : connections) {
            min_weight = std::min(min_weight, conn.weight);
            max_weight = std::max(max_weight, conn.weight);
        }
        
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        
        float layer_spacing = 4.0f;
        
        for (auto& conn : connections) {
            // Calculate positions using grid layout
            float from_y, from_z, from_x_offset;
            float to_y, to_z, to_x_offset;
            
            getNodePosition3D(conn.from_layer, conn.from_node, from_y, from_z, from_x_offset);
            getNodePosition3D(conn.to_layer, conn.to_node, to_y, to_z, to_x_offset);
            
            // Apply layer spacing along x-axis
            float from_x, to_x;
            if (conn.from_layer == 0) from_x = -layer_spacing;
            else if (conn.from_layer == 1) from_x = 0.0f;
            else from_x = layer_spacing;
            
            if (conn.to_layer == 0) to_x = -layer_spacing;
            else if (conn.to_layer == 1) to_x = 0.0f;
            else to_x = layer_spacing;
            
            // Color based on weight
            glColorFromWeight(conn.weight, min_weight, max_weight);
            
            // Highlight if hovered
            if (conn.hovered) {
                glLineWidth(4.0f);
                glColor3f(1.0f, 1.0f, 0.0f);  // Yellow for hover
            }
            
            glVertex3f(from_x, from_y, from_z);
            glVertex3f(to_x, to_y, to_z);
            
            if (conn.hovered) {
                glLineWidth(2.0f);
            }
        }
        
        glEnd();
    }
    
    void drawHoverInfo() {
        if (!hovered_connection) return;
        
        // Print hover info to console (since text rendering is complex without SDL_ttf)
        // In production, you'd render text using SDL_ttf or GLUT
        
        // Draw visual indicator on the connection (already done in drawConnections3D)
        // Additional info can be displayed here
    }
    
    Connection* findConnectionAt(int x, int y) {
        if (connections.empty()) return nullptr;
        
        // Reset all hover states
        for (auto& conn : connections) {
            conn.hovered = false;
        }
        
        // Simple hover detection: use mouse position to select a connection
        // Map mouse position to connection index using a grid-based approach
        int grid_x = (x * 10) / window_width;
        int grid_y = (y * 10) / window_height;
        int index = (grid_x + grid_y * 10) % connections.size();
        
        // Ensure index is valid
        if (index < 0 || index >= (int)connections.size()) {
            index = 0;
        }
        
        connections[index].hovered = true;
        return &connections[index];
    }
    
public:
    WeightVisualizer(const std::string& filename = "tetris_model.txt") 
        : window(nullptr), gl_context(nullptr), renderer(nullptr), model_file(filename),
          last_file_mtime(0), file_exists(false),
          window_width(1600), window_height(900), view_3d(false),
          camera_angle_x(20.0f), camera_angle_y(45.0f), camera_distance(8.0f),
          mouse_x(0), mouse_y(0), mouse_dragging(false),
          update_count(0), hovered_connection(nullptr) {
        
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return;
        }
        
        // Set OpenGL attributes
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        
        window = SDL_CreateWindow(
            "Neural Network Weight Visualizer (Press K for 3D)",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            window_width,
            window_height,
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
        
        // Create renderer for 2D mode (software renderer works better with OpenGL window)
        // We'll create it on first use in render2D()
        renderer = nullptr;
        
        // Setup OpenGL
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        
        // Setup projection
        glMatrixMode(GL_PROJECTION);
        gluPerspective(45.0, (double)window_width / window_height, 0.1, 100.0);
        glMatrixMode(GL_MODELVIEW);
        
        last_file_mtime = getFileModificationTime(model_file);
        loadModel();
    }
    
    ~WeightVisualizer() {
        if (gl_context) SDL_GL_DeleteContext(gl_context);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }
    
    void run() {
        bool running = true;
        SDL_Event event;
        
        while (running) {
            time_t current_mtime = getFileModificationTime(model_file);
            if (current_mtime != last_file_mtime && current_mtime > 0) {
                if (loadModel()) {
                    std::cout << "\n=== MODEL FILE UPDATED ===" << std::endl;
                    std::cout << "Timestamp: " << last_update_time << std::endl;
                    std::cout << "Update Count: #" << update_count << std::endl;
                    std::cout << "===========================\n" << std::endl;
                    last_file_mtime = current_mtime;
                }
            }
            
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    running = false;
                } else if (event.type == SDL_WINDOWEVENT) {
                    if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
                        window_width = event.window.data1;
                        window_height = event.window.data2;
                        if (view_3d) {
                            glViewport(0, 0, window_width, window_height);
                            glMatrixMode(GL_PROJECTION);
                            glLoadIdentity();
                            gluPerspective(45.0, (double)window_width / window_height, 0.1, 100.0);
                            glMatrixMode(GL_MODELVIEW);
                        }
                    }
                } else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q) {
                        running = false;
                    } else if (event.key.keysym.sym == SDLK_r) {
                        loadModel();
                        last_file_mtime = getFileModificationTime(model_file);
                    } else if (event.key.keysym.sym == SDLK_k) {
                        view_3d = !view_3d;
                        std::cout << "Switched to " << (view_3d ? "3D" : "2D") << " view" << std::endl;
                    }
                } else if (event.type == SDL_MOUSEMOTION) {
                    mouse_x = event.motion.x;
                    mouse_y = event.motion.y;
                    if (view_3d && mouse_dragging) {
                        camera_angle_y += event.motion.xrel * 0.5f;
                        camera_angle_x += event.motion.yrel * 0.5f;
                        camera_angle_x = std::max(-90.0f, std::min(90.0f, camera_angle_x));
                    } else if (view_3d) {
                        Connection* prev_hovered = hovered_connection;
                        hovered_connection = findConnectionAt(mouse_x, mouse_y);
                        if (hovered_connection && hovered_connection != prev_hovered) {
                            std::ostringstream oss;
                            oss << "\n[HOVER] Connection: Layer " << hovered_connection->from_layer 
                                << " Node " << hovered_connection->from_node
                                << " -> Layer " << hovered_connection->to_layer
                                << " Node " << hovered_connection->to_node
                                << " | Weight: " << std::fixed << std::setprecision(6) 
                                << hovered_connection->weight;
                            hover_text = oss.str();
                            std::cout << hover_text << std::endl;
                        }
                    }
                } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        mouse_dragging = true;
                    }
                } else if (event.type == SDL_MOUSEBUTTONUP) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        mouse_dragging = false;
                    }
                } else if (event.type == SDL_MOUSEWHEEL) {
                    if (event.wheel.y > 0) {
                        camera_distance = std::max(3.0f, camera_distance - 0.5f);
                    } else if (event.wheel.y < 0) {
                        camera_distance = std::min(15.0f, camera_distance + 0.5f);
                    }
                }
            }
            
            if (view_3d) {
                render3D();
            } else {
                render2D();
            }
            
            SDL_Delay(16);  // ~60 FPS
        }
    }
};

int main(int argc, char* argv[]) {
    std::string model_file = "tetris_model.txt";
    if (argc > 1) {
        model_file = argv[1];
    }
    
    WeightVisualizer visualizer(model_file);
    visualizer.run();
    
    return 0;
}
