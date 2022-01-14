# The OpenVINO toolkit practice

## Preparation

- Install Python 3.6/3.7/3.8 64-bit - it is needed for model downloading and converting to Intermediate Representation file (IR)
- Install Visual Studio 2015/2017
- Install latest CMake, either via following URL: https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-win64-x64.msi or via official website cmake.org
- Install the OpenVINO toolkit:
  - Download URL: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html
  - Distribution: online and offline
  - Version: 2021 4.2 LTS (This is the latest available version which is used for this sample)
  - Type: offline
  - During installation - keep all checkboxes on (Inference Engine, Model Optimizer, Open Model Zoo, OpenCV) 
- Clone this repository


## CMakelists.txt

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
find_package(OpenCV REQUIRED)
find_package(InferenceEngine 2021.4.2 REQUIRED)
```

Add directories with utilities from Open Model Zoo:
```
set(OMZ_DEMO_DIR ${InferenceEngine_DIR}/../../open_model_zoo/demos)
add_subdirectory(${OMZ_DEMO_DIR}/common/cpp/models models)
add_subdirectory(${OMZ_DEMO_DIR}/common/cpp/utils utils)
```

Create our executable file:
```
add_executable(blur_background_demo blur_background_demo.cpp)
```

Link additional libraries, such as OpenCV, Inference Engine, utility libraries from OMZ:
```
target_link_libraries(blur_background_demo ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} models utils)
```

## How to build project

We will build project using CMake.

To allow CMake find OpenVINO, we need to run 'setupvars.bat' script in command prompt for Visual Studio

```
c:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat
```

If you don't get any warnings, you can create .sln file for Visual Studio

```
cmake -B <path/for/build>
```

Then you can open generated blur_background_demo.sln in Visual Studio

## How to download pre-trained segmentation model

In same terminal (where setupvars.bat was executed) run the following command:

```
c:\blur>python "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name deeplabv3
```

Once original model is downloaded, convert it to IR
```
c:\blur>python "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader\converter.py" --name deeplabv3
```

After this you'll have the following files in c:\blur\public\deeplabv3\FP32: deeplabv3.xml, deeplabv3.bin, deeplabv3.mapping



## What about code

### What we use

We need to include some modules, like standard modules:
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

### Open video stream

Now we ready to get access to web camera. In this code camera resolution is set to 640x480, but you can specify your own if it is supported by camera.
```
	cv::VideoCapture cap;

	try {
		if (cap.open(std::stoi(input))) {
			cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
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
	auto model = std::make_unique<SegmentationModel>(modelPath, true);
	engine.SetConfig({{"CACHE_DIR", cache_dir}});

    CnnConfig cnnConfig;
    cnnConfig.devices = "CPU";
    auto execNetwork = model->loadExecutableNetwork(cnnConfig, engine);
```

Get input and output layer (we use model with single input and output):
```
	std::string inputName  = model->getInputsNames()[0];
	std::string outputName = model->getOutputsNames()[0];
```

Create inference request (later we will use it to launch our model):
```
	InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
```

    
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
			outFrame = remove_background(frame, segmentationResult->asRef<ImageResult>());
			break;
		case BACKGROUND:
			outFrame = remove_background(frame, background, segmentationResult->asRef<ImageResult>());
			break;
		case BLUR:
			outFrame = blur_background(frame, segmentationResult->asRef<ImageResult>());
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
cv::Mat remove_background(cv::Mat frame, cv::Mat background, const SegmentationResult& segmentationResult)
```
Because size of frame and background could differ, we should make them equal:
``` 
	cv::resize(background, background, frame.size());
```
The segmentation model we used, has multiple classes, not only person. But we not interested in other classes. So we need to get rid of all other classes except person. We need to know id of our class (`15` for suggested model) which can help perform masking.
```
	const int personLabel = 15;
	cv::Mat personMask = cv::Mat(mask.size(), mask.type(), personLabel);
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