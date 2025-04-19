/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "common.hpp"
#include "formatable.hpp"
#include "match.hpp"

#include <opencv2/core.hpp>
#include <vector>

// C-compatible struct definitions for Python ctypes
extern "C" {

// Config struct for C API
typedef struct {
    float nxcorr_threshold;
    float subpixel_step;
    float min_variance;
    int mode; // 0 = LIMITED, 1 = FULL
#ifdef BICOS_CUDA
    int precision; // 0 = SINGLE, 1 = DOUBLE
#endif
    int variant_type; // 0 = NoDuplicates, 1 = Consistency
    int max_lr_diff; // Only used if variant_type = 1
    int no_dupes; // Only used if variant_type = 1
} BicosConfig;

// Result struct for C API
typedef struct {
    void* disparity_data;
    int disparity_rows;
    int disparity_cols;
    int disparity_type;
    void* corrmap_data;
    int corrmap_rows;
    int corrmap_cols;
    int corrmap_type;
} BicosResult;

// Function to convert C config to C++ config
BICOS::Config convertConfig(const BicosConfig* c_config) {
    BICOS::Config config;
    
    if (c_config->nxcorr_threshold >= 0) {
        config.nxcorr_threshold = c_config->nxcorr_threshold;
    }
    
    if (c_config->subpixel_step >= 0) {
        config.subpixel_step = c_config->subpixel_step;
    }
    
    if (c_config->min_variance >= 0) {
        config.min_variance = c_config->min_variance;
    }
    
    config.mode = (c_config->mode == 0) ? 
        BICOS::TransformMode::LIMITED : BICOS::TransformMode::FULL;
    
#ifdef BICOS_CUDA
    config.precision = (c_config->precision == 0) ? 
        BICOS::Precision::SINGLE : BICOS::Precision::DOUBLE;
#endif
    
    if (c_config->variant_type == 0) {
        config.variant = BICOS::Variant::NoDuplicates{};
    } else {
        BICOS::Variant::Consistency consistency;
        consistency.max_lr_diff = c_config->max_lr_diff;
        consistency.no_dupes = (c_config->no_dupes != 0);
        config.variant = consistency;
    }
    
    return config;
}

// Function to create default config
BicosConfig* BICOS_CreateDefaultConfig() {
    BicosConfig* config = new BicosConfig();
    
    // Set default values
    config->nxcorr_threshold = 0.5f;
    config->subpixel_step = -1.0f; // Use -1 to indicate null/not set
    config->min_variance = -1.0f;  // Use -1 to indicate null/not set
    config->mode = 0; // LIMITED
#ifdef BICOS_CUDA
    config->precision = 0; // SINGLE
#endif
    config->variant_type = 0; // NoDuplicates
    config->max_lr_diff = 1;
    config->no_dupes = 0;
    
    return config;
}

// Function to free config
void BICOS_FreeConfig(BicosConfig* config) {
    delete config;
}

// Function to free result
void BICOS_FreeResult(BicosResult* result) {
    delete result;
}

// Helper function to convert image data from Python to C++
BICOS::Image convertImageData(void* data, int rows, int cols, int type) {
    cv::Mat header(rows, cols, type, data);
#ifdef BICOS_CUDA
    return BICOS::Image(header);
#else
    return header;
#endif
}

// Main matching function
BicosResult* BICOS_Match(
    void** stack0_data, int* stack0_rows, int* stack0_cols, int* stack0_types, int stack0_size,
    void** stack1_data, int* stack1_rows, int* stack1_cols, int* stack1_types, int stack1_size,
    BicosConfig* config
) {
    try {
        std::vector<BICOS::Image> stack0_, stack1_;
        BICOS::Image disparity_, corrmap_;
        
        // Convert stack0
        for (int i = 0; i < stack0_size; i++) {
            stack0_.push_back(convertImageData(
                stack0_data[i], stack0_rows[i], stack0_cols[i], stack0_types[i]
            ));
        }
        
        // Convert stack1
        for (int i = 0; i < stack1_size; i++) {
            stack1_.push_back(convertImageData(
                stack1_data[i], stack1_rows[i], stack1_cols[i], stack1_types[i]
            ));
        }
        
        // Convert config
        BICOS::Config cpp_config = convertConfig(config);
        
        // Perform matching
        BICOS::match(stack0_, stack1_, disparity_, cpp_config, &corrmap_);
        
        // Create result
        BicosResult* result = new BicosResult();
        
        // Copy disparity
        cv::Mat disparity_cpu;
#ifdef BICOS_CUDA
        disparity_.download(disparity_cpu);
#else
        disparity_cpu = disparity_;
#endif
        result->disparity_rows = disparity_cpu.rows;
        result->disparity_cols = disparity_cpu.cols;
        result->disparity_type = disparity_cpu.type();
        
        // Allocate memory for disparity data
        size_t disparity_size = disparity_cpu.total() * disparity_cpu.elemSize();
        result->disparity_data = malloc(disparity_size);
        memcpy(result->disparity_data, disparity_cpu.data, disparity_size);
        
        // Copy corrmap
        cv::Mat corrmap_cpu;
#ifdef BICOS_CUDA
        corrmap_.download(corrmap_cpu);
#else
        corrmap_cpu = corrmap_;
#endif
        result->corrmap_rows = corrmap_cpu.rows;
        result->corrmap_cols = corrmap_cpu.cols;
        result->corrmap_type = corrmap_cpu.type();
        
        // Allocate memory for corrmap data
        size_t corrmap_size = corrmap_cpu.total() * corrmap_cpu.elemSize();
        result->corrmap_data = malloc(corrmap_size);
        memcpy(result->corrmap_data, corrmap_cpu.data, corrmap_size);
        
        return result;
    } catch (const BICOS::Exception& e) {
        // Handle exceptions
        return nullptr;
    }
}

// Function to get invalid disparity value
float BICOS_InvalidDisparityFloat() {
    return BICOS::INVALID_DISP<float>;
}

int16_t BICOS_InvalidDisparityInt16() {
    return BICOS::INVALID_DISP<int16_t>;
}

} // extern "C"
