// Copyright (C) 2009 - 2014 NICTA
//
// Authors:
// - Vikas Reddy (vikas.reddy at ieee dot org, rvikas2333 at gmail dot com)
//
// This file is provided without any warranty of
// fitness for any purpose. You can redistribute
// this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)


#ifndef ANOMALY_DETECTION_PROTO_HPP_
#define ANOMALY_DETECTION_PROTO_HPP_

#include "inc.h"

enum state
{
	undefined, busy_state, idle_state, complete
};

struct gbr_config
{

	s32 kernel_size;
	Col<u32> theta_arr;
	field<cv::Mat> kernel_mtx;
	float variance;
	float lambda;
	float phase;
};


struct edgefilt_config
{
	s32 kernel_size;
	u32 n_kernels;
	field<cv::Mat> kernel_mtx;
};

class step_val
{
	s32 busy_cnt;
	s32 idle_cnt;
	state status;
public:
	friend class AnomalyDetection;
	step_val(s32 val1, s32 val2)
	{
		busy_cnt = val1;
		idle_cnt = val2;
	}
	step_val()
	{
	}
	;
	void init(s32 val1, s32 val2, state current_state)
	{
		busy_cnt = val1;
		idle_cnt = val2;
		status = current_state;
	}
};

typedef struct debug_mb
{
	cv::Point coords;
	vec fv;
	double p_value;
};

class AnomalyDetection
{

public:
	vector<debug_mb> loc;
	field<Mat<double> > training_dist;
	Mat<double> ad_image;

private:
	field<vector<step_val> > pixel_process;
	field<umat> t_on_off;
	field<mat> t_on_off_freq;
	field<vec> global_features;
	field<mat> local_features;
	field<mat> blk_occ;
	field<vec> test_feature;
	field<vec> model_prms;
	field<vec> tmprl_win_feature;
	field<vec> tmprl_win_blkocc;
	field<mat> tmprl_win_edges;

	cube bw_array;

	field<vector<vec> > fr_edge_hist_tr;
	field<vector<vec> > fr_edge_hist_tst;
	field<mat> tmprl_edge_hist_tr;

	field<vec> cur_edge_hist;
	mat smoothing_filter;

	vector<Mat<u32> > input_binary_masks;
	vector<Mat<u32> > test_binary_masks;

	Col<double> DCT_coeffs;
	Mat<double> dct_mtx;

	bool first_frame;
	u32 height, width;
	u32 train_frame_cnt, test_frames, test_frame_cnt;
	u32 temporal_window;
	step_val curr_run;
	u32 N, ovlstep;
	u32 xmb, ymb, no_of_blks;
	u32 testfrm_idx, total_blks;
	cv::Mat estimated_ad_frame;

	uvec sorted_gaussians;

	gbr_config gbr_prms;
	edgefilt_config edgefilt_prms;
	field<cv::Mat> edge_frames;
	field<vector<vec> > rep_textures_tr;
	field<mat> rep_textures;

	double speed_thr, size_thr, hist_thr, ofset, size_bw, speed_bw;

	/*optical flow related*/

	cv::Mat cur_frame, prev_frame;
	Mat<double> mv_mtx_x, mv_mtx_y;
	field<vec> cur_avgblk_mv;
	field<vector<vec> > fr_avgblk_mv_tr;
	field<vector<vec> > fr_avgblk_mv_tst;
	field<mat> tmprl_avgblk_mv_tr;

	/*kernel density related*/
	field<vec> spd_pdf, sze_pdf;
	double kd_min[2], kd_max[2], n_kd_eval_pts[2];
	vec spd_kd_eval_pts, sze_kd_eval_pts;

	field<mat> tmprl_avgblk_mv;

public:
	AnomalyDetection(const cv::Mat &frame, const u32 &training_len, const rowvec &cur_params);
	AnomalyDetection();
	void collect_frames_for_training(const cv::Mat &frame, const cv::Mat &grysc_frame);
	void collect_frames_for_testing(const cv::Mat &frame, const cv::Mat &grysc_frame);
	void train_model();
	void test(const cv::Mat &frame, const cv::Mat &grysc_frame);
	void
	compute_subwindows_features(const cv::Mat &in_frame, umat &out_arma_frame);
	void compute_subwindows_edges(const cv::Mat &in_frame, const cv::Mat &in_bin_frame, const u32 &fidx);
	void extract_spatio_temporal_features(const u32 &fs_idx, const u32 &stage);
	void trace_block(cv::Mat &frame, const int i, const int j);
	void create_dct_table(int N);
	void trace_block_color(cv::Mat &frame, const int i, const int j);

	void display_feature_value(u32 fidx);
	void compute_feature_vector(const Col<double> &input_vector, Col<double> &out_fv);

	void gabor_filter(const cv::Mat &in_frame);
	void gabor_kernel_initialisation();

	void edge_filter(const cv::Mat &in_frame);
	void edge_filter_initialisation();

	void estimate_motion(const cv::Mat &bin_frame, const cv::Mat &grysc_frame, const umat &sub_win_mtx);
	double find_8connected_neighbours(const u32 &x, const u32 &y, const umat &blk_occ_mtx);

	void kernel_density_estimate(const mat & training_samples, vec & pdf, const u32 &idx);
	virtual ~AnomalyDetection();

	void save_ad_model_params();
	void load_ad_model_params();
	void numtostr(int num, char *str);

	double test_for_speed(const Col<double> &query_vec, const u32 &x, const u32 &y);

	double test_for_size(const Col<double> &query_vec, const u32 &x, const u32 &y);

	double test_for_texture(const mat &model, const Col<double> &query_vec);

	inline double measure_correlation_coefficient(const Col<double> &a, const Col<double> &b);

	void online_texture_quantisation(const mat &texture_mat, mat &rep_textures);

};

#endif /* ANOMALY_DETECTION_PROTO_HPP */
