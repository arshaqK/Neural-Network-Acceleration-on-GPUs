#include <fstream>
#include <vector>
#include <cassert>
#include <iostream>

void parseImageData(const std::string &path, std::vector<float> &data, int total) {
    std::ifstream stream(path, std::ios::binary);
    assert(stream.is_open());

    int32_t header = 0, count = 0, height = 0, width = 0;
    stream.read((char*)&header, 4);
    stream.read((char*)&count, 4);
    stream.read((char*)&height, 4);
    stream.read((char*)&width, 4);

    header = __builtin_bswap32(header);
    count = __builtin_bswap32(count);
    height = __builtin_bswap32(height);
    width = __builtin_bswap32(width);

    assert(header == 2051);
    assert(count >= total);

    data.resize(total * height * width);
    for (int i = 0; i < total * height * width; ++i) {
        unsigned char val = 0;
        stream.read((char*)&val, sizeof(val));
        data[i] = val / 255.0f;
    }
}

void parseLabelData(const std::string &path, std::vector<float> &data, int total) {
    std::ifstream stream(path, std::ios::binary);
    assert(stream.is_open());

    int32_t header = 0, count = 0;
    stream.read((char*)&header, 4);
    stream.read((char*)&count, 4);

    header = __builtin_bswap32(header);
    count = __builtin_bswap32(count);

    assert(header == 2049);
    assert(count >= total);

    data.resize(total * 10, 0.0f);
    for (int i = 0; i < total; ++i) {
        unsigned char val = 0;
        stream.read((char*)&val, sizeof(val));
        data[i * 10 + val] = 1.0f;
    }
}

int main() {
    const int NEURONS_IN = 784;
    const int NEURONS_MID = 128;
    const int NEURONS_OUT = 10;
    const int SAMPLES_TRAINING = 60000;
    const int SAMPLES_TESTING = 10000;
    const int SIZE_BATCH = 128;
    const int NUM_EPOCHS = 5;

    std::vector<float> dataTrainImg, dataTrainLbl;
    std::vector<float> dataTestImg, dataTestLbl;

    parseImageData("../../data/train-images.idx3-ubyte", dataTrainImg, SAMPLES_TRAINING);
    parseLabelData("../../data/train-labels.idx1-ubyte", dataTrainLbl, SAMPLES_TRAINING);
    parseImageData("../../data/t10k-images.idx3-ubyte", dataTestImg, SAMPLES_TESTING);
    parseLabelData("../../data/t10k-labels.idx1-ubyte", dataTestLbl, SAMPLES_TESTING);

    NeuralNet *model;
    cudaMallocManaged(&model, sizeof(NeuralNet));
    initNet(model, NEURONS_IN, NEURONS_MID, NEURONS_OUT, SIZE_BATCH);

    std::vector<float> batchX(SIZE_BATCH * NEURONS_IN);
    std::vector<float> batchY(SIZE_BATCH * NEURONS_OUT);

    for (int e = 0; e < NUM_EPOCHS; ++e) {
        std::cout << "Epoch " << e + 1 << " / " << NUM_EPOCHS << "\n";

        for (int i = 0; i < SAMPLES_TRAINING; i += SIZE_BATCH) {
            int currentBatchSize = std::min(SIZE_BATCH, SAMPLES_TRAINING - i);

            std::copy(dataTrainImg.begin() + i * NEURONS_IN, 
                     dataTrainImg.begin() + (i + currentBatchSize) * NEURONS_IN, 
                     batchX.begin());
            std::copy(dataTrainLbl.begin() + i * NEURONS_OUT, 
                     dataTrainLbl.begin() + (i + currentBatchSize) * NEURONS_OUT, 
                     batchY.begin());

            cudaMemcpy(model->d_batch_input, batchX.data(), 
                      currentBatchSize * NEURONS_IN * sizeof(float), 
                      cudaMemcpyHostToDevice);
            cudaMemcpy(model->d_batch_d_output, batchY.data(), 
                      currentBatchSize * NEURONS_OUT * sizeof(float), 
                      cudaMemcpyHostToDevice);

            cudaForward(model);
            cudaBackward(model, 0.01f);
        }
    }

    std::cout << "Training complete!\n";
    return 0;
}