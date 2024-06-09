#include <iostream>
#include"util.h"

using namespace cv;
using namespace std;

Mat linspace(float begin, float finish, int number)
{
	float interval = (finish - begin) / (number - 1);//  
	Mat f(1, number, CV_32FC1);
	for (int i = 0; i < f.rows; i++) 
	{
		for (int j = 0; j < f.cols; j++) 
		{
			f.at<float>(i, j) = begin + j * interval;
		}
	}
	return f;
}

void get_norm_rigid_mesh_inv_grid(Mat& grid, Mat& W_inv, const int input_height, const int input_width, const int grid_h, const int grid_w)
{
	float interval_x = input_width / grid_w;
	float interval_y = input_height / grid_h;
	const int h = grid_h + 1;
	const int w = grid_w + 1;
	const int length = h * w;
	Mat norm_rigid_mesh(length, 2, CV_32FC1);
	///norm_rigid_mesh.create(length, 2, CV_32FC1);
	Mat W(length + 3, length + 3, CV_32FC1);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const int row_ind = i * w + j;
			const float x = (j * interval_x)*2.0 / float(input_width) - 1.0;
			const float y = (i * interval_y)*2.0 / float(input_height) - 1.0;
			
			W.at<float>(row_ind, 0) = 1;
			W.at<float>(row_ind, 1) = x;
			W.at<float>(row_ind, 2) = y;

			W.at<float>(length, 3 + row_ind) = 1;
			W.at<float>(length + 1, 3 + row_ind) = x;
			W.at<float>(length + 2, 3 + row_ind) = y;

			norm_rigid_mesh.at<float>(row_ind, 0) = x;
			norm_rigid_mesh.at<float>(row_ind, 1) = y;
		}
	}
	for (int i = 0; i < length; i++)
	{
		for (int j = 0;j < length; j++)
		{
			const float d2_ij = powf(W.at<float>(i, 0) - W.at<float>(j, 0), 2.0) + powf(W.at<float>(i, 1) - W.at<float>(j, 1), 2.0) + powf(W.at<float>(i, 2) - W.at<float>(j, 2), 2.0);
			W.at<float>(i, 3 + j) = d2_ij * logf(d2_ij + 1e-6);
		}
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			W.at<float>(length + i, j) = 0;
		}
	}

	W_inv = W.inv();

	interval_x = 2.0 / (input_width - 1);
	interval_y = 2.0 / (input_height - 1);
	const int grid_width = input_height * input_width;

	///Mat grid(length + 3, grid_width, CV_32FC1);
	grid.create(length + 3, grid_width, CV_32FC1);
	for (int i = 0; i < input_height; i++)
	{
		for (int j = 0; j < input_width; j++)
		{
			const float x = -1.0 + j * interval_x;
			const float y = -1.0 + i * interval_y;
			const int col_ind = i * input_width + j;
			grid.at<float>(0, col_ind) = 1;
			grid.at<float>(1, col_ind) = x;
			grid.at<float>(2, col_ind) = y;
		}
	}
	
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < grid_width; j++)
		{
			const float d2_ij = powf(norm_rigid_mesh.at<float>(i, 0) - grid.at<float>(1, j), 2.0) + powf(norm_rigid_mesh.at<float>(i, 1) - grid.at<float>(2, j), 2.0);
			grid.at<float>(3 + i, j) = d2_ij * logf(d2_ij + 1e-6);
		}
	}
	norm_rigid_mesh.release();
}

void get_ori_rigid_mesh_tp(Mat& tp, Mat& ori_mesh_np_x, Mat& ori_mesh_np_y, const float* offset, const int input_height, const int input_width, const int grid_h, const int grid_w)
{
	const float interval_x = input_width / grid_w;
	const float interval_y = input_height / grid_h;
	const int h = grid_h + 1;
	const int w = grid_w + 1;
	const int length = h * w;
	tp.create(length + 3, 2, CV_32FC1);
	ori_mesh_np_x.create(h, w, CV_32FC1);
	ori_mesh_np_y.create(h, w, CV_32FC1);
	
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const int row_ind = i * w + j;
			const float x = j * interval_x + offset[row_ind * 2];
			const float y = i * interval_y + offset[row_ind * 2 + 1];
			tp.at<float>(row_ind, 0) = (j * interval_x + offset[row_ind * 2])*2.0 / float(input_width) - 1.0;
			tp.at<float>(row_ind, 1) = (i * interval_y + offset[row_ind * 2+1])*2.0 / float(input_height) - 1.0;
			
			ori_mesh_np_x.at<float>(i, j) = x;
			ori_mesh_np_y.at<float>(i, j) = y;
		}

	}
	for (int i = 0; i < 3; i++)
	{
		tp.at<float>(length + i, 0) = 0;
		tp.at<float>(length + i, 1) = 0;
	}
}

Mat _interpolate(Mat im, Mat xy_flat, Size out_size)  ////xy_flat的形状是(2, 65536)
{
	const int height = im.size[2];
	const int width = im.size[3];
	const int max_x = width - 1;
	const int max_y = height - 1;
	const float height_f = float(height);
	const float width_f = float(width);
	const int area = height * width;
	const float* pdata = (float*)im.data;   ////形状是(1,3,256,256)
	Mat output(out_size.height, out_size.width, CV_32FC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			const int col_ind = i * width + j;
			float x = (xy_flat.at<float>(0, col_ind) + 1.0)*width_f*0.5;
			float y = (xy_flat.at<float>(1, col_ind) + 1.0)*height_f*0.5;

			int x0 = int(x);
			int x1 = x0 + 1;
			int y0 = int(y);
			int y1 = y0 + 1;
			x0 = std::min(std::max(x0, 0), max_x);
			x1 = std::min(std::max(x1, 0), max_x);
			y0 = std::min(std::max(y0, 0), max_y);
			y1 = std::min(std::max(y1, 0), max_y);

			int base_y0 = y0 * width;
			int base_y1 = y1 * width;
			int idx_a = base_y0 + x0;
			int idx_b = base_y1 + x0;
			int idx_c = base_y0 + x1;
			int idx_d = base_y1 + x;

			float x0_f = float(x0);
			float x1_f = float(x1);
			float y0_f = float(y0);
			float y1_f = float(y1);
			float wa = (x1_f - x) * (y1_f - y);
			float wb = (x1_f - x) * (y - y0_f);
			float wc = (x - x0_f) * (y1_f - y);
			float wd = (x - x0_f) * (y - y0_f);

			float pix_r = wa * pdata[idx_a] + wb * pdata[idx_b] + wc * pdata[idx_c] + wd * pdata[idx_d];
			float pix_g = wa * pdata[area + idx_a] + wb * pdata[area + idx_b] + wc * pdata[area + idx_c] + wd * pdata[area + idx_d];
			float pix_b = wa * pdata[2 * area + idx_a] + wb * pdata[2 * area + idx_b] + wc * pdata[2 * area + idx_c] + wd * pdata[2 * area + idx_d];
			
			output.at<Vec3f>(i, j) = Vec3f(pix_r, pix_g, pix_b);
		}
	}
	return output;
}

Mat draw_mesh_on_warp(const Mat warp, const Mat f_local_x, const Mat f_local_y)
{
	const int height = warp.rows;
	const int width = warp.cols;
	const int grid_h = f_local_x.rows - 1;
	const int grid_w = f_local_x.cols - 1;

	double minValue_x, maxValue_x;    // 最大值，最小值
	cv::Point  minIdx_x, maxIdx_x;    // 最小值坐标，最大值坐标     
	cv::minMaxLoc(f_local_x, &minValue_x, &maxValue_x, &minIdx_x, &maxIdx_x);
	const int min_w = int(std::min(minValue_x, 0.0));
	const int max_w = int(std::max(maxValue_x, double(width)));

	double minValue_y, maxValue_y;    // 最大值，最小值
	cv::Point  minIdx_y, maxIdx_y;    // 最小值坐标，最大值坐标     
	cv::minMaxLoc(f_local_y, &minValue_y, &maxValue_y, &minIdx_y, &maxIdx_y);
	const int min_h = int(std::min(minValue_y, 0.0));
	const int max_h = int(std::max(maxValue_y, double(height)));
	
	const int cw = max_w - min_w;
	const int ch = max_h - min_h;
	const int pad_top = 0 - min_h + 5;
	const int pad_bottom = ch + 10 - (pad_top + height);
	const int pad_left = 0 - min_w + 5;
	const int pad_right = cw + 10 - (pad_left + width);
	Mat pic;
	copyMakeBorder(warp, pic, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, Scalar(255,255,255));
	pic.convertTo(pic, CV_8UC3);
	for (int i = 0; i < (grid_h + 1); i++)
	{
		for (int j = 0; j < (grid_w + 1); j++)
		{
			if (j == grid_w && i == grid_h) continue;
			else if (j == grid_w)
			{
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i + 1, j) - min_w + 5), int(f_local_y.at<float>(i + 1, j) - min_h + 5)), Scalar(0, 255, 0), 2);
			}
			else if (i == grid_h)
			{
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i, j + 1) - min_w + 5), int(f_local_y.at<float>(i, j + 1) - min_h + 5)), Scalar(0, 255, 0), 2);
			}
			else
			{
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i + 1, j) - min_w + 5), int(f_local_y.at<float>(i + 1, j) - min_h + 5)), Scalar(0, 255, 0), 2);
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i, j + 1) - min_w + 5), int(f_local_y.at<float>(i, j + 1) - min_h + 5)), Scalar(0, 255, 0), 2);
			}
		}
	}
	return pic;
}