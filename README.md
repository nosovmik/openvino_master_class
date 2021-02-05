# The OpenVINO toolkit master-class

## Preparation

- Install Visual Studio 2015/2017
- Install latest CMake, either via following URL: https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-win64-x64.msi or via official website cmake.org
- Install the OpenVINO toolkit https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html
- Clone this repository
- Clone Open Model Zoo [repository](

## How build project

We will build project using CMake.

Setting minimum requirements:
```
cmake_minimum_required (VERSION 3.10)
```

Name our project:
```
project(blur_background_demo)
```

Lets find necessary packages OpenCV and Inference Engine
```
find_package(OpenCV  REQUIRED)
add_definitions(-DUSE_OPENCV)
find_package(InferenceEngine 2.0 REQUIRED)
```

Add directories with utilities from Open Model Zoo (the OMZ_DEMO_DIR value we will set later):
```
add_subdirectory(${OMZ_DEMO_DIR}/common/cpp/models models)
add_subdirectory(${OMZ_DEMO_DIR}/common/cpp/utils utils)
add_subdirectory(${OMZ_DEMO_DIR}/thirdparty/gflags gflags EXCLUDE_FROM_ALL)
```

Create our executable file:
```
add_executable(blur_background_demo blur_background_demo.cpp)
```

Link additional libraries, such as OpenCV, Inference Engine, utility libraries from OMZ:
```
target_link_libraries(blur_background_demo ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} models utils gflags)
```

Then we can build our project using command (now we set OMZ_DEMO_DIR value with path to `<open_model_zoo>/demos`):
```
cmake -B <path/for/build> -DOMZ_DEMO_ZOO <path/to/omz>
```

## What about code

### What we use

We need to include some modules, like standart modules:
```
#include <iostream>
#include <string>
```
then OpenCV and Inference Engine:
```
#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>
```
and modules from Open Model Zoo utilities:
```
#include <models/segmentation_model.h>
```
for segmentation model wrapper
```
#include <utils/ocv_common.hpp>
```
for common utilities
```
#include <utils/performance_metrics.hpp>
```
for fps counter

### Application parameters

We will use as little parameters as possible. We need to specify:
- Camera id 
- Path to background image
- Path to model

To simplify our code we won't use any utilities for command line parser and use `*argv[]` argument instead:
```
	std::string input = argv[1];
	std::string backgroundPath = argv[2];
	std::string model_path = argv[3];
```

### Open video stream

Now we ready to get access to web camera. In this code camera resolution is set to 1280x720, but you can specify your own if it is supported by camera. 
```
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
	catch (const std::invalid_argument&) {}
	catch (const std::out_of_range&) {}
```

### Prepare engine and model

First we create engine:
```
	InferenceEngine::Core engine;
```
Then create model wrapper and read network from disk:
```
	ModelBase *model = new SegmentationModel(model_path, true);
	InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork(model->getModelFileName());
```
Get input and output layer (we use model with single input and output):
```
	model->prepareInputsOutputs(cnnNetwork);

	std::string inputName  = model->getInputsNames()[0];
	std::string outputName = model->getOutputsNames()[0];
```
Load model to device (in this example we will load model to GPU, but could also load it to CPU):
```
	InferenceEngine::ExecutableNetwork execNetwork = engine.LoadNetwork(cnnNetwork, "GPU");
```
Create inference request (later we will use it to launch our model):
```
	InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
```


	   
	cv::Mat background = cv::imread(backgroundPath);

	PerformanceMetrics metrics;

	int type = DELETE;
    
### Main cycle

Now we ready to use our model with video from camera. We will do this in cycle while camera is on.
```
    while (cap.isOpened())
    {
        ...
    }
```
Since we want to see fps, we need know time of each frame process:
```

		auto startTime = std::chrono::steady_clock::now();
```
Now we ready ro read first frame from camera:
```
		cv::Mat frame;
		cap.read(frame);
```
Then prepare it for Inference Engine request (the function `wrapMat2Blob` is defined in <utils/ocv_common.hpp> and perform necessary transformation from cv::Mat to IE supported format):
```
		InferenceEngine::Blob::Ptr imgBlob = wrapMat2Blob(frame);
		inferRequest.SetBlob(inputName, imgBlob);
```
After setting data we can launch our model and get result:
```
		inferRequest.Infer();
		InferenceEngine::Blob::Ptr result = inferRequest.GetBlob(outputName);
```


		InferenceResult inferenceResult;
		inferenceResult.outputsData.emplace(outputName, 
			std::make_shared<InferenceEngine::TBlob<float>>(*InferenceEngine::as<InferenceEngine::TBlob<float>>(result)));
		inferenceResult.internalModelData = std::shared_ptr<InternalImageModelData>(new InternalImageModelData(frame.size[1], frame.size[0]));

Then model gives us result, it often need some postprocessing (e.g. resize it to frame size). Class `SegmentationModel` already has it and we can use it.
```
		std::unique_ptr<ResultBase> segmentationResult = model->postprocess(inferenceResult);
```	
Now we ready to process our frame with segmentation mask and perform necessary transformation. The code below has few different transformation types:
```
		cv::Mat outFrame;
		switch (type)
		{
		case DELETE:
			outFrame = remove_background(frame, segmentationResult->asRef<SegmentationResult>());
			break;
		case BACKGROUND:
			outFrame = remove_background(frame, background, segmentationResult->asRef<SegmentationResult>());
			break;
		case BLUR:
			outFrame = blur_background(frame, segmentationResult->asRef<SegmentationResult>());
			break;
		default:
			break;
		}
```
We perform almost all action with this frame and ready to show, but before it we need to update our metrics and add fps value on our frame:
```
		metrics.update(startTime, outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
		cv::imshow("Video", outFrame);
```
Last but not least, lets add some commands, like exit application when `ESC` pressed, or switch transformation type by `TAB` key:
```
		int key = cv::waitKey(1);
		if (key == 27)
			break;
		if (key == 9)
		{
			type++;
			if (type == NONE)
				type = 0;
		}
```
That's all.

### What about image transformations

Above we note few transformation functions, now we define one of them, `replace_background`. This function will find a person on the `frame` using `mask` and replace `background`.
```
cv::Mat remove_background(cv::Mat frame, cv::Mat background, SegmentationResult& segmentationResult)
```
Because size of frame and background could differ, we should make them equal:
``` 
	cv::resize(background, background, frame.size());
```
The segmentation model we used, has multiple classes, not only person. But we not interested in other classes. So we need to get rid of all other classes except person. We need to know id of our class (`15` for suggested model) which can help perform masking.
```
	const int personLabel = 15;
	cv::Mat personMask = cv::Mat(mask.size(), mask.type(), 15);
	cv::compare(mask, personMask, personMask, cv::CMP_EQ);
```
After it we can mask our frame and higlight only person:
```
	cv::Mat maskedFrame;
	cv::bitwise_or(frame, frame, maskedFrame, personMask);
```
Next we perform similar operation with `background` image, but with opposite mask:
```
    cv::Mat backgroundMask;
	cv::bitwise_not(personMask, backgroundMask);
	cv::Mat maskedBackground;
	cv::bitwise_or(background, background, maskedBackground, backgroundMask);
```
Then after concatenation two masked image we get result:
```
	cv::bitwise_or(maskedFrame, maskedBackground, frame);

	return frame;
```