# ifndef UTILS
# define UTILS
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat linspace(float begin, float finish, int number);
void get_norm_rigid_mesh_inv_grid(cv::Mat& grid, cv::Mat& W_inv, const int input_height, const int input_width, const int grid_w, const int grid_h);
void get_ori_rigid_mesh_tp(cv::Mat& tp, cv::Mat& ori_mesh_np_x, cv::Mat& ori_mesh_np_y, const float* offset, const int input_height, const int input_width, const int grid_h, const int grid_w);
cv::Mat _interpolate(cv::Mat im, cv::Mat xy_flat, cv::Size out_size);
cv::Mat draw_mesh_on_warp(const cv::Mat warp, const cv::Mat f_local_x, const cv::Mat f_local_y);


#endif
