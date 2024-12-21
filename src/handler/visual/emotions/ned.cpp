#include "ned.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/face.hpp>

using namespace cv;
using namespace cv::face;

inline std::string NED::detectEmotions(VideoCapture camera) {
	std::string label;
	bool firstFrame = false;
	CascadeClassifier faceDetector;
	Mat frame, gray, lastframe;
	std::vector<Rect> faces;
	std::vector<std::vector<Point2f>> landmarks;

	std::string cascadeName = "data/haarcascade_frontalface_alt2.xml";
	std::string modelName = "data/lbfmodel.yml";
	Ptr<Facemark> facemark; 

	try {
		faceDetector.load(cascadeName);
		facemark->loadModel(modelName);
		facemark = FacemarkLBF::create();
	} catch (std::exception&) {
		return "[!] Runtime Error: Failed to load Cascade or Facemark Model";
	}

	// TODO: Alter this later to hook to a unique pointer
	if (!camera.isOpened()) {
		try {
			camera.open(0);
		} catch (std::exception&) {
			return "[!] Runtime Error: Failed to connect to a camera";
		}
	}

	// TODO: Patch in CWE-120/CWE-20 prevention
	while (camera.read(frame)) {
		if (frame.empty()) {
			break;
		}

		if (frame.cols > 1920 || frame.rows > 1080) {
			break; // We don't want to overload the CPU on wide-screen processing
			// TODO: Implement a GPU handler to wide-screen data
		}

		// ImageSmoothing::smooth(frame); // TODO: Uncomment this when the smoother/deblur is built

		// Clear old data to prevent misreads
		faces.clear();
		landmarks.clear();


		cvtColor(frame, gray, COLOR_BGR2GRAY); // Convert the colors to grayscale,
		// This helps with landmark detection, though not entirely necessary for other models

		if (firstFrame) {
			lastframe = gray.clone();
			firstFrame = false;
		}

		faceDetector.detectMultiScale(gray, faces);

		if (facemark->fit(gray, faces, landmarks)) {
			label = NED::getEmotion(
				gray, lastframe, faceDetector, 
				landmarks, faces, facemark
			);
		} else {
			label = "No known emotions detected...";
		}

		lastframe = gray.clone();
	}

	return label;
};

inline float NED::getDistance(
	const Point2f& pointA, const Point2f& pointB
) {
	// (Ax - Bx)^2 + (Ay - By)^2
	return sqrt(
		powf((
			pointA.x - pointB.x
		), 2)
	) + powf((
		pointA.y - pointB.y
	), 2);
};

inline std::string NED::getEmotion(
	InputArray frame, InputArray lastFrame,
	CascadeClassifier classifier,
	const std::vector<std::vector<Point2f>>& landmarks,
	const std::vector<Rect>& faces, const Ptr<Facemark>& facemark
) {
	std::string result;
	std::vector<std::vector<Point2f>> lastLandmarks;
	std::vector<Rect> lastFaces;

	classifier.detectMultiScale(lastFrame, lastFaces);
	facemark->fit(lastFrame, lastFaces, lastLandmarks);

	std::vector<float> distTo33(landmarks[0].size());
	std::vector<float> lastDistTo33(lastLandmarks[0].size());


	// Parallel run the for loop to allow for a quicker calculation of landmarks
	#pragma omp parallel for
	for (int i = 0; i < landmarks[0].size(); i++) {
		distTo33[i] = getDistance(landmarks[0][i], landmarks[0][33]);
		lastDistTo33[i] = getDistance(lastLandmarks[0][i], lastLandmarks[0][33]);
	}

	std::vector<Condition> conditions = {
		{
			#pragma region Surprise
			[&]() {
				return dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
					dist17to36 > 0 && dist26to45 > 0 && // AU 2
					dist33to38 > 0 && dist33to37 > 0 && dist33to43 > 0 && dist33to44 > 0  && // AU 5
					(
						dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0 && // AU 26
						dist8to33 > 0 && dist7to33 > 0 && dist9to33 > 0
					); // AU 27
			},
			"Surprise"
			#pragma endregion
		},
		{
			#pragma region Sadness
			[&]() {
				return dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
					dist21to22 < 0 && dist20to23 < 0 && // AU 4
					dist5to48 < 0 && dist11to54 < 0; // AU 15
			},
			"Sadness"
			#pragma endregion
		},
		{
			#pragma region Fear
			[&]() {
				return dist21to33 > 0 && dist22to33 > 0 && dist20to33 > 0 && dist23to33 > 0 && // AU 1
					dist17to36 > 0 && dist26to45 > 0 && // AU 2
					dist21to22 < 0 && dist20to23 < 0 && // AU 4
					dist33to37 > 0 && dist33to38 > 0 && dist33to43 > 0 && dist33to44 > 0 && // AU 5
					(
						dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0 && // AU 26
						dist8to33 > 0 && dist7to33 > 0 && dist9to33 > 0
					); // AU 27
			},
			"Fear"
			#pragma endregion
		},
		{
			#pragma region Disgust
			[&]() {
				return dist21to22 < 0 && dist20to23 < 0 && dist21to33 < 0 && dist22to33 < 0 && // AU 9
					dist33to56 > 0 && dist33to57 > 0 && dist33to58 > 0 && // AU 16
					(
						(dist5to48 < 0 && dist11to54 < 0) ||
						(dist61to67 > 0 && dist62to66 > 0 && dist63to65 > 0)
					); // AU 15 or AU 26
			},
			"Disgust"
			#pragma endregion
		},
		{
			#pragma region Angry
			[&]() {
				return dist21to22 < 0 && dist20to23 < 0 && // AU 4
					dist33to38 > 0 && dist33to37 > 0 && dist33to43 > 0 && dist33to44 > 0 &&// AU 5
					dist27to40 < 0 && dist27to47 < 0 && // AU 7
					((dist61to67 < 0 && dist62to66 < 0 && dist63to65 < 0 ) || // AU 23 or AU 24
					(dist50to61 < 0 && dist51to62 < 0 && dist52to63 < 0));
			},
			"Angry"
			#pragma endregion
		},
		{
			#pragma region Happiness
			[&]() {
				return dist36to48 < 0 && dist45to54 < 0;
			},
			"Happiness"
			#pragma endregion
		},
		{
			#pragma region Neutral
			[&]() {
				return true;
			},
			"Neutral"
			#pragma endregion
		},
		{
			#pragma region None
			[&]() {
				return false;
			},
			"No emotion detected"
			#pragma endregion
		}
	};

	#pragma omp parallel for
	for (const auto& condition : conditions) {
		if (condition.check()) {
			result = condition.result;
			break;
		}
	}

	return result;
}