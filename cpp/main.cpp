#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include"util.h"
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace std;
using namespace dnn;


class RecRecNet
{
public:
	RecRecNet(string model_path);
	vector<Mat> detect(Mat srcimg);
private:
	const int input_height = 256;
	const int input_width = 256;
	const int grid_h = 8;
	const int grid_w = 8;
	Mat grid;
	Mat W_inv;
	Net net;
};

RecRecNet::RecRecNet(string model_path)
{
	this->net = readNet(model_path);
	get_norm_rigid_mesh_inv_grid(this->grid, this->W_inv, this->input_height, this->input_width, this->grid_h, this->grid_w);
}

vector<Mat> RecRecNet::detect(Mat srcimg)
{
	Mat img;
	resize(srcimg, img, Size(this->input_width, this->input_height), INTER_LINEAR);
	img.convertTo(img, CV_32FC3, 1.0 / 127.5, -1.0);
	Mat blob = blobFromImage(img);

	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   
	const float* offset = (float*)outs[0].data;
	Mat tp, ori_mesh_np_x, ori_mesh_np_y;
	get_ori_rigid_mesh_tp(tp, ori_mesh_np_x, ori_mesh_np_y, offset, this->input_height, this->input_width, this->grid_h, this->grid_w);
	Mat T = W_inv * tp;   ////_solve_system
	T = T.t();    ////ÉáÆúbatchsize

	Mat T_g = T * this->grid;

	Mat output_tps = _interpolate(blob, T_g, Size(this->input_width, this->input_height));
	Mat rectangling_np = (output_tps + 1)*127.5;
	rectangling_np.convertTo(rectangling_np, CV_8UC3);
	Mat input_np = (img + 1)*127.5;

	vector<Mat> outputs;
	outputs.emplace_back(rectangling_np);
	outputs.emplace_back(input_np);
	outputs.emplace_back(ori_mesh_np_x);
	outputs.emplace_back(ori_mesh_np_y);

	return outputs;
}


int main()
{
	RecRecNet mynet("model_deploy.onnx");
	string imgpath = "testimgs/10.jpg";
	Mat srcimg = imread(imgpath);

	vector<Mat> outputs = mynet.detect(srcimg);
	Mat input_with_mesh = draw_mesh_on_warp(outputs[1], outputs[2], outputs[3]);


	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	namedWindow("rect", WINDOW_NORMAL);
	imshow("rect", outputs[0]);
	namedWindow("mesh", WINDOW_NORMAL);
	imshow("mesh", input_with_mesh);
	waitKey(0);
	destroyAllWindows();
}