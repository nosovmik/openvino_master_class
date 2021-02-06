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
const int NONE = 2;

cv::Mat replace_background(cv::Mat frame, cv::Mat background, SegmentationResult& segmentationResult)
{
	auto mask = segmentationResult.mask;

	cv::resize(background, background, frame.size());

	const int personLabel = 15;
	cv::Mat personMask = cv::Mat(mask.size(), mask.type(), 15);
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


cv::Mat remove_background(cv::Mat frame, SegmentationResult& segmentationResult)
{
	auto mask = segmentationResult.mask;


	const int personLabel = 15;
	cv::Mat personMask = cv::Mat(mask.size(), mask.type(), 15);
	cv::compare(mask, personMask, personMask, cv::CMP_EQ);

	cv::Mat maskedFrame;
	cv::bitwise_or(frame, frame, maskedFrame, personMask);

	return maskedFrame;
}

cv::Mat blur_background(cv::Mat frame, SegmentationResult& segmentationResult)
{
	auto mask = segmentationResult.mask;

	const int personLabel = 15;
	cv::Mat personMask = cv::Mat(mask.size(), mask.type(), 15);
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

	std::string input = argv[1];
	std::string backgroundPath = argv[2];
	std::string model_path = argv[3];

	cv::VideoCapture cap;

	try {
		if (cap.open(std::stoi(input))) {
			cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
			cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
			cap.set(cv::CAP_PROP_AUTOFOCUS, true);
			cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
		}
	}
	catch (const std::invalid_argument&) {} // If stoi conversion failed, let's try another way to open capture device
	catch (const std::out_of_range&) {}

	InferenceEngine::Core engine;

	ModelBase *model = new SegmentationModel(model_path, true);
	InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork(model->getModelFileName());

	model->prepareInputsOutputs(cnnNetwork);

	std::string inputName  = model->getInputsNames()[0];
	std::string outputName = model->getOutputsNames()[0];

	InferenceEngine::ExecutableNetwork execNetwork = engine.LoadNetwork(cnnNetwork, "GPU");

	InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
	   
	cv::Mat background = cv::imread(backgroundPath);

	PerformanceMetrics metrics;

	int type = DELETE;

	while (cap.isOpened())
	{
		auto startTime = std::chrono::steady_clock::now();

		cv::Mat frame;
		cap.read(frame);

		InferenceEngine::Blob::Ptr imgBlob = wrapMat2Blob(frame);
		inferRequest.SetBlob(inputName, imgBlob);

		inferRequest.Infer();
		InferenceEngine::Blob::Ptr result = inferRequest.GetBlob(outputName);

		InferenceResult inferenceResult;
		inferenceResult.outputsData.emplace(outputName, 
			std::make_shared<InferenceEngine::TBlob<float>>(*InferenceEngine::as<InferenceEngine::TBlob<float>>(result)));
		inferenceResult.internalModelData = std::shared_ptr<InternalImageModelData>(new InternalImageModelData(frame.size[1], frame.size[0]));

		std::unique_ptr<ResultBase> segmentationResult = model->postprocess(inferenceResult);
		

		cv::Mat outFrame;
		switch (type)
		{
		case DELETE:
			outFrame = remove_background(frame, segmentationResult->asRef<SegmentationResult>());
			break;
		case BACKGROUND:
			outFrame = replace_background(frame, background, segmentationResult->asRef<SegmentationResult>());
			break;
		case BLUR:
			outFrame = blur_background(frame, segmentationResult->asRef<SegmentationResult>());
			break;
		default:
			break;
		}
		
		metrics.update(startTime, outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
		cv::imshow("Video", outFrame);
		int key = cv::waitKey(1);
		if (key == 27)
			break;
		if (key == 9)
		{
			type++;
			if (type == NONE)
				type = 0;
		}
	}
};