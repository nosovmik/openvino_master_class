#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

#include <models/segmentation_model.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>

const int DELETE = 0;
const int BACKGROUND = 1;
const int BLUR = 2;

const int personLabel = 15;  // Masked background will have '15' value for pixel belonging to person

cv::Mat replace_background(cv::Mat frame, cv::Mat background, const ImageResult& segmentationResult)
{
    auto mask = segmentationResult.resultImage;

    cv::resize(background, background, frame.size());

    cv::Mat personMask = cv::Mat(mask.size(), mask.type(), personLabel);
    cv::compare(mask, personMask, personMask, cv::CMP_EQ);

    cv::Mat maskedFrame;
    cv::bitwise_or(frame, frame, maskedFrame, personMask);

    cv::Mat backgroundMask;
    cv::bitwise_not(personMask, backgroundMask);
    cv::Mat maskedBackground;
    cv::bitwise_or(background, background, maskedBackground, backgroundMask);

    cv::bitwise_or(maskedFrame, maskedBackground, frame);

    return frame;
}


cv::Mat remove_background(const cv::Mat& frame, const ImageResult& segmentationResult)
{
    auto mask = segmentationResult.resultImage;

    cv::Mat personMask = cv::Mat(mask.size(), mask.type(), personLabel);
    cv::compare(mask, personMask, personMask, cv::CMP_EQ);

    cv::Mat maskedFrame;
    cv::bitwise_or(frame, frame, maskedFrame, personMask);

    return maskedFrame;
}

cv::Mat blur_background(cv::Mat frame, const ImageResult& segmentationResult)
{
    auto mask = segmentationResult.resultImage;

    cv::Mat personMask = cv::Mat(mask.size(), mask.type(), personLabel);
    cv::compare(mask, personMask, personMask, cv::CMP_EQ);

    cv::Mat maskedFrame;
    cv::bitwise_or(frame, frame, maskedFrame, personMask);

    cv::Mat backgroundMask;
    cv::bitwise_not(personMask, backgroundMask);
    cv::Mat maskedBackground;
    cv::bitwise_or(frame, frame, maskedBackground, backgroundMask);

    cv::Mat blurredBackground;
    cv::blur(maskedBackground, blurredBackground, cv::Size(21, 21));

    cv::bitwise_or(maskedFrame, blurredBackground, frame);

    return frame;
}

int main(int argc, char *argv[])
{
    // Some hard-coded values. Change it to your paths (or get it from cmd args)
    int cameraIndex = 0;
    std::string backgroundPath = "c:\\blur\\test_background.jpg";
    std::string modelPath = "c:\\blur\\public\\deeplabv3\\FP32\\deeplabv3.xml";
    std::string cacheDir = "c:\\blur\\cache";
    std::string device = "GPU";

    cv::VideoCapture cap;

    try {
        if (cap.open(cameraIndex)) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap.set(cv::CAP_PROP_AUTOFOCUS, true);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        } else {
            std::cout << "Camera is not opened, try some another value\n";
            return -1;
        }
    }
    catch (std::exception& ex) {
        std::cout << "Failed to open camera " << ex.what() << std::endl;
        return -1;
    }

    InferenceEngine::Core engine;
    engine.SetConfig({{"CACHE_DIR", cacheDir}});

    auto model = std::make_unique<SegmentationModel>(modelPath, true);
    CnnConfig cnnConfig;
    cnnConfig.devices = device;
    auto execNetwork = model->loadExecutableNetwork(cnnConfig, engine);

    std::string inputName  = model->getInputsNames()[0];
    std::string outputName = model->getOutputsNames()[0];

    InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
       
    cv::Mat background = cv::imread(backgroundPath);

    PerformanceMetrics metrics;

    int type = DELETE;

    while (cap.isOpened())
    {
        cv::Mat frame;
        cap.read(frame);

        InferenceEngine::Blob::Ptr imgBlob = wrapMat2Blob(frame);
        inferRequest.SetBlob(inputName, imgBlob);

        inferRequest.Infer();
        InferenceEngine::Blob::Ptr result = inferRequest.GetBlob(outputName);

        auto result_mem = std::dynamic_pointer_cast<InferenceEngine::MemoryBlob>(result);

        InferenceResult inferenceResult;
        inferenceResult.outputsData.emplace(outputName, result_mem);
        inferenceResult.internalModelData = std::make_shared<InternalImageModelData>(frame.size[1], frame.size[0]);

        std::unique_ptr<ResultBase> segmentationResult = model->postprocess(inferenceResult);
        

        cv::Mat outFrame;
        switch (type)
        {
        case DELETE:
            outFrame = remove_background(frame, segmentationResult->asRef<ImageResult>());
            break;
        case BACKGROUND:
            outFrame = replace_background(frame, background, segmentationResult->asRef<ImageResult>());
            break;
        case BLUR:
            outFrame = blur_background(frame, segmentationResult->asRef<ImageResult>());
            break;
        default:
            break;
        }

        metrics.update(std::chrono::steady_clock::now(), outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
        cv::imshow("Video", outFrame);
        int key = cv::waitKey(1);
        if (key == 27)
            break;
        if (key == 9)
        {
            type++;
            if (type > BLUR)
                type = 0;
        }
    }
};