#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#include "../../csrc/engine.h"


using namespace std;
using namespace cv;


float max(float a, float b) {
    return a > b ? a : b;
}


float min(float a, float b) {
    return b < a ? b : a;
}


class ImagesIds {
public:

    ImagesIds(string file_name) : image_names(nullptr), image_ids(nullptr), len(-1) {
        ifstream myfile;
        myfile.open(file_name, ios::in);
        len = 0;
        string line;
        while (getline(myfile, line)) {
            len++;
        }
        myfile.close();
        
        image_names = new string[len];
        image_ids = new int[len];
        myfile.open(file_name, ios::in);
        int i = 0;
        while (getline(myfile, line)) {
            int space_position = line.find(" ");
            int image_id = stoi(line.substr(0, space_position));
            string image_name = line.substr(space_position+1, line.length());
            image_names[i] = image_name;
            image_ids[i] = image_id;
            i++;
        }
    }

    int dump() {
        for (int i = 0; i < len; i++) {
            cout << image_ids[i] << " " << image_names[i] << endl;
        }
    }

    ~ImagesIds() {
        if (image_names != nullptr) delete[] image_names;
        if (image_ids != nullptr) delete[] image_ids;
    }

    string* image_names;
    int* image_ids;
    int len;
};


class CUDA_Buffers {
public:

    CUDA_Buffers(int num_det, const vector<int> &input_size) : data_d(nullptr), scores_d(nullptr), boxes_d(nullptr), classes_d(nullptr) {
	    cudaMalloc(&data_d, 3 * input_size[0] * input_size[1] * sizeof(float));
	    cudaMalloc(&scores_d, num_det * sizeof(float));
	    cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
	    cudaMalloc(&classes_d, num_det * sizeof(float));
    }

    ~CUDA_Buffers() {
    }

    void *data_d;
    void *scores_d;
    void *boxes_d;
    void *classes_d;
};


class InferenceResults {
public:
    
    InferenceResults(int num_det) : scores(nullptr), boxes(nullptr), classes(nullptr), len(num_det) {
        scores = new float[num_det];
	    boxes = new float[num_det * 4];
	    classes = new float[num_det];
    }

    ~InferenceResults() {
        if (scores != nullptr) delete[] scores;
        if (boxes != nullptr) delete[] boxes;
        if (classes != nullptr) delete[] classes;
    }

    float *scores;
    float *boxes;
    float *classes;
    int len;
};


retinanet::Engine& load_engine(const string &engine_file) {
    auto *engine = new retinanet::Engine(engine_file);
    return *engine;
}


vector<int>& get_engine_input_size(retinanet::Engine &engine) {
    vector<int> hw = engine.getInputSize();
    vector<int> *wh = new vector<int> {hw[1], hw[0]};
    return *wh;
}


vector<float>& prepare_image(const string &image_file, const vector<int> &input_size, vector<int> *original_size = nullptr) {
    auto image = imread(image_file, IMREAD_COLOR);
    if (original_size != nullptr) {
        original_size->clear();
        original_size->push_back(image.size().width);
        original_size->push_back(image.size().height);
    }
    cv::resize(image, image, Size(input_size[0], input_size[1]));
    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

    int channels = 3;
    vector<float> img;
    vector<float> *data = new vector<float>(channels * input_size[0] * input_size[1]);

    if (pixels.isContinuous())
        img.assign((float*)pixels.datastart, (float*)pixels.dataend);
    else {
        cerr << "Error reading image " << image_file << endl;
        return *data;
    }

    vector<float> mean {0.485, 0.456, 0.406};
    vector<float> std {0.229, 0.224, 0.225};

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = input_size[0] * input_size[1]; j < hw; j++) {
            (*data)[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
        }
    }
    return *data;
}


InferenceResults& inference_engine(retinanet::Engine &engine, const CUDA_Buffers &buffers, const vector<float> &data) {
    // Copy image to device
    size_t dataSize = data.size() * sizeof(float);
    cudaMemcpy(buffers.data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

    // Run inference
    vector<void *> buffs = {buffers.data_d, buffers.scores_d, buffers.boxes_d, buffers.classes_d};
    engine.infer(buffs, 1);

    // Get back the bounding boxes
    int num_det = engine.getMaxDetections();
    InferenceResults *results = new InferenceResults(num_det);
    cudaMemcpy(results->scores, buffers.scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->boxes, buffers.boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(results->classes, buffers.classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
    return *results;
}


int resize_boxes(InferenceResults &results, const vector<int> &input_size, const vector<int> &original_size) {
    float x_scale = 1. * original_size[0] / input_size[0];
    float y_scale = 1. * original_size[1] / input_size[1];
    for (int i = 0; i < results.len; i++) {
        results.boxes[i*4+0] = results.boxes[i*4+0] * x_scale;
        results.boxes[i*4+1] = results.boxes[i*4+1] * y_scale;
        results.boxes[i*4+2] = results.boxes[i*4+2] * x_scale;
        results.boxes[i*4+3] = min(results.boxes[i*4+3] * y_scale, original_size[1]);
    }
    return 0;
}


int predict(retinanet::Engine &engine, const CUDA_Buffers &buffers, const ImagesIds &images_and_ids, const string &out_file) {
    int num_det = engine.getMaxDetections();
    vector<int> input_size = get_engine_input_size(engine);
    ofstream myfile;
    myfile.open(out_file, ios::out | ios::trunc);
    for (int i = 0; i < images_and_ids.len; i++) {
        cout << "\r" << i+1 << " out of " << images_and_ids.len << flush;
        myfile << images_and_ids.image_ids[i] << ' ' << images_and_ids.image_names[i] << endl;
        vector<int> original_size;
        vector<float> data = prepare_image(images_and_ids.image_names[i], input_size, &original_size);
        InferenceResults results = inference_engine(engine, buffers, data);
        resize_boxes(results, input_size, original_size);
        for (int j = 0; j < num_det; j++) {
            myfile << results.scores[j] << ' ' << results.boxes[j*4+0] << ' ' << results.boxes[j*4+1] << ' ' << results.boxes[j*4+2] << ' ' << results.boxes[j*4+3] << ' ' << results.classes[j] << endl;
        }
    }
    cout << endl;
    myfile.close();
    return 0;
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "Usage : ./predict images_and_ids.txt engine.plan detections.txt" << endl;
        return 0;
    }
    ImagesIds images_and_ids(argv[1]);
    retinanet::Engine engine = load_engine(argv[2]);
    int num_det = engine.getMaxDetections();
    vector<int> input_size = get_engine_input_size(engine);
    CUDA_Buffers buffers(num_det, input_size);
    predict(engine, buffers, images_and_ids, argv[3]);
    return 0;
}
