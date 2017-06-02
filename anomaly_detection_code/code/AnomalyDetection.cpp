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


#include "inc.h"
#include "AnomalyDetection.h"


AnomalyDetection::AnomalyDetection(const cv::Mat &frame, const u32 &training_len, const rowvec &cur_params)
{
	height = frame.rows;
	width = frame.cols;
	N = cur_params(1);
	ovlstep = cur_params(2);
	xmb = (width - N) / ovlstep;
	ymb = (height - N) / ovlstep;

	total_blks = (1 + ymb) * (1 + xmb);
	pixel_process.set_size((1 + ymb), (1 + xmb));
	t_on_off.set_size((1 + ymb), (1 + xmb));
	t_on_off_freq.set_size((1 + ymb), (1 + xmb));
	train_frame_cnt = 0;
	test_frame_cnt = 0;
	test_frames = 0;
	testfrm_idx = 0;
	temporal_window = cur_params(3);
	global_features.set_size(total_blks);
	local_features.set_size((1 + ymb), (1 + xmb));
	blk_occ.set_size((1 + ymb), (1 + xmb));
	model_prms.set_size((1 + ymb), (1 + xmb));
	test_feature.set_size((1 + ymb), (1 + xmb));
	first_frame = true;
//	gm_local_model.set_size((1 + ymb), (1 + xmb));

	ad_image.set_size(height, width);

	mv_mtx_x.set_size(height, width);
	mv_mtx_y.set_size(height, width);

	training_dist.set_size(2);
	for (u32 i = 0; i < training_dist.n_elem; ++i)
	{
		training_dist(i).set_size((1 + ymb), (1 + xmb));
		training_dist(i).zeros();
	}

	no_of_blks = (1 + ymb) * (1 + xmb);
	bw_array.set_size((1 + ymb), (1 + xmb), 2);
	cur_edge_hist.set_size((1 + ymb), (1 + xmb));
	fr_edge_hist_tr.set_size((1 + ymb), (1 + xmb));
	fr_edge_hist_tst.set_size((1 + ymb), (1 + xmb));
	tmprl_edge_hist_tr.set_size((1 + ymb), (1 + xmb));
	//	rep_textures_tr.set_size((1 + ymb), (1 + xmb));
	rep_textures.set_size((1 + ymb), (1 + xmb));

	cur_avgblk_mv.set_size((1 + ymb), (1 + xmb));
	fr_avgblk_mv_tr.set_size((1 + ymb), (1 + xmb));
	fr_avgblk_mv_tst.set_size((1 + ymb), (1 + xmb));
	tmprl_avgblk_mv_tr.set_size((1 + ymb), (1 + xmb));
	tmprl_avgblk_mv.set_size((1 + ymb), (1 + xmb));

	spd_pdf.set_size((1 + ymb), (1 + xmb));
	sze_pdf.set_size((1 + ymb), (1 + xmb));

	cout << " MB x: " << (1 + xmb) << " MB y: " << (1 + ymb) << endl;

	tmprl_win_feature.set_size((1 + ymb), (1 + xmb));
	tmprl_win_edges.set_size((1 + ymb), (1 + xmb));
	tmprl_win_blkocc.set_size((1 + ymb), (1 + xmb));

	Mat<double> bw_vals(1 + ymb, 2);
	double ofset1, ofset2;
	u32 n_edge_bins = 4;

	smoothing_filter.set_size(3, 3);

	smoothing_filter << 1 << 2 << 1 << endr << 2 << 4 << 2 << endr << 1 << 2 << 1 << endr;

	smoothing_filter = smoothing_filter / accu(smoothing_filter);
	//		smoothing_filter.print_trans("filter: ");
	//
	//		exit(0);

	size_thr = cur_params(4);
	speed_thr = cur_params(5);
	hist_thr = cur_params(6);
	ofset = cur_params(7);
	size_bw = cur_params(8);
	speed_bw = cur_params(9);




	bw_array.slice(0).fill(size_bw);
	bw_array.slice(1).fill(speed_bw);

	ofset1 = 0;
	ofset2 = 0;

	n_kd_eval_pts[0] = 30;
	kd_max[0] = 15;
	kd_min[0] = 0;

	n_kd_eval_pts[1] = 25;
	kd_max[1] = 256;
	kd_min[1] = 0;

	spd_kd_eval_pts = linspace<vec> (kd_min[0], kd_max[0], n_kd_eval_pts[0] + 1);
	sze_kd_eval_pts = linspace<vec> (kd_min[1], kd_max[1], n_kd_eval_pts[1] + 1);


	for (u32 y = 0; y < (1 + ymb); ++y)
	{

		//		cout << bw_vals(y, 0) << " " << bw_vals(y, 1) << endl;
		for (u32 x = 0; x < (1 + xmb); ++x)
		{
			tmprl_win_feature(y, x).set_size(temporal_window);
			tmprl_win_feature(y, x).fill(0);

			tmprl_win_blkocc(y, x).set_size(temporal_window);
			tmprl_win_blkocc(y, x).fill(0);

			tmprl_win_edges(y, x).set_size(n_edge_bins, temporal_window);
			tmprl_win_edges(y, x).zeros();

			model_prms(y, x).set_size(1);
			model_prms(y, x).fill(0);

			bw_array.slice(0)(y, x) = max(bw_array.slice(0)(y, x), 8.0) + ofset1;
			bw_array.slice(1)(y, x) = max(bw_array.slice(1)(y, x), 8.0) + ofset2;

			cur_edge_hist(y, x).set_size(n_edge_bins);
			cur_avgblk_mv(y, x).set_size(2);

			tmprl_avgblk_mv(y, x).set_size(2, temporal_window);
			tmprl_avgblk_mv(y, x).zeros();

			spd_pdf(y, x).set_size(n_kd_eval_pts[0] + 1);
			spd_pdf(y, x).zeros();

			sze_pdf(y, x).set_size(n_kd_eval_pts[1] + 1);
			sze_pdf(y, x).zeros();

		}

		ofset1 += 0;
		ofset2 += 0;
	}

	bw_array.slice(0).fill(size_bw);
	bw_array.slice(1).fill(speed_bw);

	//	bw_array.slice(0).print("size");
	//	bw_array.slice(1).print("speed");
	//					exit(0);

	int t_N = temporal_window;

	dct_mtx = ones<arma::Mat<double> > (t_N, t_N);
	DCT_coeffs.set_size(t_N * t_N);

	create_dct_table(t_N);

	//	//INITIALISE THE DCT MATRIX
	for (int i = 0; i < t_N; i++)
	{
		for (int j = 0; j < t_N; j++)
		{
			dct_mtx(i, j) = DCT_coeffs(i * t_N + j);
		}
	}

	cv::Size imsize(width, height);
	estimated_ad_frame = cv::Mat::zeros(imsize, CV_8UC1);

	/*******************************Gabor filter initialisations *******************************/
	gbr_prms.kernel_size = 9;
	gbr_prms.theta_arr.set_size(n_edge_bins);
	gbr_prms.kernel_mtx.set_size(gbr_prms.theta_arr.n_elem);

	for (u32 i = 0; i < gbr_prms.theta_arr.n_elem; ++i)
	{
		gbr_prms.theta_arr(i) = (u32) (180.0 / (gbr_prms.theta_arr.n_elem)) * i;
		gbr_prms.kernel_mtx(i) = cv::Mat::zeros(cv::Size(gbr_prms.kernel_size, gbr_prms.kernel_size), CV_32FC1);

	}

	gbr_prms.phase = (float) 90 * CV_PI / 180;
	gbr_prms.lambda = 3;
	gbr_prms.variance = 9;

	edge_frames.set_size(gbr_prms.theta_arr.n_elem);

	for (u32 i = 0; i < gbr_prms.theta_arr.n_elem; ++i)
	{
		edge_frames(i) = cv::Mat::zeros(imsize, CV_32FC1);
	}

	gabor_kernel_initialisation();
	/*******************************Gabor filter initialisations *******************************/

	/*******************************edge filter initialisations *******************************/
	edgefilt_prms.n_kernels = 4;
	edgefilt_prms.kernel_size = 3;
	edgefilt_prms.kernel_mtx.set_size(edgefilt_prms.n_kernels);

	for (u32 i = 0; i < edgefilt_prms.n_kernels; ++i)
	{
		edgefilt_prms.kernel_mtx(i) = cv::Mat::zeros(cv::Size(edgefilt_prms.kernel_size, edgefilt_prms.kernel_size),
				CV_32FC1);
	}
	edge_filter_initialisation();
	/*******************************edge filter initialisations *******************************/

	ofstream features_file;
	features_file.open("output/mtx.txt", ios::out);
	features_file.close();

	//		exit(0);

}

AnomalyDetection::~AnomalyDetection()
{
	// TODO Auto-generated destructor stub
}

void AnomalyDetection::train_model()
{

	umat arma_frame((1 + ymb), (1 + xmb));
	step_val tmp;

	ofstream features_file, log_file;
	features_file.open("output/mtx1.txt", ios::out);
	features_file.close();
	log_file.open("output/log.txt", ios::out);

	train_frame_cnt = input_binary_masks.size();

	if (temporal_window > train_frame_cnt)
	{
		cout << "temporal window greater than total training frames" << endl;
		exit(-23);
	}
	u32 temporal_win_cnt = train_frame_cnt - temporal_window;
	u32 mid_pt = (temporal_window) / 2;

	global_features.set_size(total_blks * temporal_win_cnt);

	cout << xmb << " " << ymb << " " << temporal_win_cnt << endl;
	cout << cur_edge_hist(0, 0).n_elem << endl;

	//allocation of memory and setting the temporal window sizes

	for (u32 x = 0; x < (1 + xmb); ++x)
	{
		for (u32 y = 0; y < (1 + ymb); ++y)
		{
			local_features(y, x).set_size(temporal_win_cnt, 1);
			blk_occ(y, x).set_size(temporal_win_cnt, 1);
			tmprl_edge_hist_tr(y, x).set_size(gbr_prms.theta_arr.n_elem, temporal_win_cnt);
			tmprl_avgblk_mv_tr(y, x).set_size(2, temporal_win_cnt);

			//			for (u32 idx = 0; idx < temporal_win_cnt; ++idx)
			//			{
			//				tmprl_edge_hist_tr(y, x)(idx).set_size(
			//						gbr_prms.theta_arr.n_elem	);
			//			}

		}
	}

	cout << "training start" << endl;

	for (u32 f_idx = 0; f_idx < (train_frame_cnt - temporal_window); f_idx++)
	{

		for (u32 tw_idx = f_idx, z = 0; tw_idx < (f_idx + temporal_window); ++tw_idx, ++z)
		{
			arma_frame = input_binary_masks[tw_idx];

			for (u32 x = 0; x < (1 + xmb); ++x)
			{
				for (u32 y = 0; y < (1 + ymb); ++y)
				{

					tmprl_win_feature(y, x)(z) = find_8connected_neighbours(x, y, arma_frame);
					//					tmprl_win_feature(y, x)(z) = arma_frame(y, x);

					tmprl_win_blkocc(y, x)(z) = arma_frame(y, x);

					tmprl_win_edges(y, x).col(z) = fr_edge_hist_tr(y, x)[tw_idx];

					tmprl_avgblk_mv(y, x).col(z) = fr_avgblk_mv_tr(y, x)[tw_idx];

				}
			}

		}

		Col<double> fv(1);
		// compute DCT of each region over the temporal window
		for (u32 x = 0; x < (1 + xmb); ++x)
		{
			for (u32 y = 0; y < (1 + ymb); ++y)
			{
				compute_feature_vector(tmprl_win_feature(y, x), fv);
				local_features(y, x).row(f_idx) = trans(fv);

				compute_feature_vector(tmprl_win_blkocc(y, x), fv);
				blk_occ(y, x).row(f_idx) = trans(fv);

				tmprl_edge_hist_tr(y, x).col(f_idx) = (tmprl_win_edges(y, x).col(mid_pt));
				tmprl_avgblk_mv_tr(y, x).col(f_idx) = sum(tmprl_avgblk_mv(y, x), 1);

			}
		}

	}

	cout << "training end" << endl;

	//copy the whole features into a container

	mat tmp_x((1 + ymb), (1 + xmb));
	mat tmp_y((1 + ymb), (1 + xmb));

	tmp_x.zeros();
	tmp_y.zeros();

	for (u32 y = 0; y < (1 + ymb); ++y)
	{
		for (u32 x = 0; x < (1 + xmb); ++x)
		{

			u32 cnt = 0, cnt1 = 0;
			mat dist(1, 1);
			mat l1_norm_mv(temporal_win_cnt, 1);
			mat size_ftr(temporal_win_cnt, 1);
			l1_norm_mv.zeros();
			size_ftr.zeros();

			for (u32 t = 0; t < temporal_win_cnt; ++t)
			{
				dist(0, 0) = norm(tmprl_avgblk_mv_tr(y, x).col(t), 1);

				if (dist(0, 0) > 0)
				{
					l1_norm_mv.row(cnt++) = dist.row(0);
				}

				if (local_features(y, x).row(t)(0) > 0)
				{
					size_ftr.row(cnt1++) = local_features(y, x).row(t);
				}

			}

			if ((cnt > 0) && (cnt < temporal_win_cnt))
			{
				l1_norm_mv.shed_rows(cnt, l1_norm_mv.n_elem - 1);
			}

			kernel_density_estimate(l1_norm_mv, spd_pdf(y, x), 0);

			if ((cnt1 > 0) && (cnt1 < temporal_win_cnt))
			{
				size_ftr.shed_rows(cnt1, size_ftr.n_elem - 1);
			}

			kernel_density_estimate(size_ftr, sze_pdf(y, x), 1);
			//
			//				if (x == 12 && y == 0)
			//				{
			//					cout << "................." << endl;
			//					sze_pdf(y, x).print_trans("size pdf");
			//					local_features(y, x).print_trans(" zero pdf block loc");
			//					exit(0);
			//				}

			//reset the vectors
			fr_avgblk_mv_tr(y, x).clear();
			fr_edge_hist_tr(y, x).clear();

			mat valid_edge_hist_samples(tmprl_edge_hist_tr(y, x).n_rows, temporal_win_cnt);
			valid_edge_hist_samples.zeros();
			cnt = 0;

			// for expt and debugging purpose
			for (u32 z = 0; z < temporal_win_cnt; ++z)
			{

				if (accu(blk_occ(y, x).row(z)) > 8)
				{
					valid_edge_hist_samples.col(cnt++) = tmprl_edge_hist_tr(y, x).col(z);
				}

				tmp_x(y, x) += tmprl_avgblk_mv_tr(y, x).col(z)(0);
				tmp_y(y, x) += tmprl_avgblk_mv_tr(y, x).col(z)(1);

			}

			u32 size = max(cnt, u32(2));
			mat edge_hist_samples(tmprl_edge_hist_tr(y, x).n_rows, size);

			for (u32 z = 0; z < size; ++z)
			{
				edge_hist_samples.col(z) = valid_edge_hist_samples.col(z);
			}

#if 0
			field<vec> edge_hist_samples(size);
			// initialisation of at least two samples for k-means rountine
			edge_hist_samples(0) = valid_edge_hist_samples.col(0);
			edge_hist_samples(1) = valid_edge_hist_samples.col(0);

			for (u32 z = 0; z < cnt; ++z)
			{
				edge_hist_samples(z) = valid_edge_hist_samples.col(z);
			}

			if (false)
			if (x == 5 && y == 6)
			{
				cout << ".........." << endl;
				gm_local_model(y, x).train_kmeans(edge_hist_samples, 2, 5, 0.9, false, false);
				cout << gm_local_model(y, x).means << endl;
				cout << gm_local_model(y, x).dcovs << endl;
				cout << gm_local_model(y, x).weights << endl;

				exit(-1);
			}
			//			cout << cnt << "  " << size << endl;
			//			cout << valid_edge_hist_samples << endl;

			log_file << cnt << "  " << size << endl;
			log_file << valid_edge_hist_samples << endl;

			//			exit(0);
			u32 n_gaussians = 3;
			gm_local_model(y, x).train_kmeans(edge_hist_samples, n_gaussians, 5, 0.9, false, false);

			for (u32 z = 0; z < n_gaussians; ++z)
			{

				//							cout << valid_edge_hist_samples << endl;

				log_file << gm_local_model(y, x).means << endl;
				log_file << gm_local_model(y, x).dcovs << endl;
			}

			log_file << " .........." << endl;
			//			if (edge_hist_samples.n_elem < n_gaussians)
			//			{
			//				n_gaussians = edge_hist_samples.n_elem / 2;
			//
			//				//				cout << edge_hist_samples.n_elem << " " << n_gaussians << endl;
			//				gm_local_model(y, x).train_kmeans(edge_hist_samples, n_gaussians, 5, 0.9, false, false);
			//			}
			//			else
			//			{
			//				gm_local_model(y, x).train_kmeans(edge_hist_samples, n_gaussians, 5, 0.9, false, false);
			//			}
			//
#endif
			//			cout << edge_hist_samples << endl;
			//			cout << tmprl_edge_hist_tr(y, x) << endl;
			//			cout << "......" << endl;

			online_texture_quantisation(edge_hist_samples, rep_textures(y, x));

			//			if (x == 6 && y == 5)
			//			{
			//				exit(0);
			//			}

			edge_hist_samples.reset();

		}
	}

	u32 lmax = 0;
	for (u32 y = 0; y < (1 + ymb); ++y)
	{
		for (u32 x = 0; x < (1 + xmb); ++x)
		{
			lmax = max(lmax, (u32) rep_textures(y, x).n_cols);

		}
	}

	//	rep_textures.save("rep_textures");
	cout << "max val: " << lmax << endl;
	log_file.close();
	save_ad_model_params();

}
//convert open cv frame to armadillo matrix format
void AnomalyDetection::compute_subwindows_features(const cv::Mat &in_frame, umat &out_arma_frame)
{

	mat tmp_mtx(1, 1);
	umat tmp_arma_frame(height, width);
	u32 x = 0, y = 0;
	//copy frame into arma structure
	for (u32 i = 0; i < height; i++)
	{
		for (u32 j = 0; j < width; j++)
		{

			tmp_arma_frame(i, j) = in_frame.at<arma::u8> (i, j) & 0x1;

		}
	}
	//analyse region and take a call to classify it as idle/busy block
	for (u32 i = 0; i <= height - N; i += ovlstep)
	{
		for (u32 j = 0; j <= width - N; j += ovlstep)
		{
			tmp_mtx = sum(sum(tmp_arma_frame.submat(i, j, (i + N - 1), (j + N - 1))));

			out_arma_frame(y, x) = tmp_mtx(0, 0);

			x++;
		}
		x = 0;
		y++;
	}

}

void AnomalyDetection::compute_subwindows_edges(const cv::Mat &in_frame, const cv::Mat &in_bin_frame, const u32 &fidx)
{

	mat tmp_mtx(1, 1);
	mat tmp_arma_frame = zeros<mat> (height, width);
	u32 x = 0, y = 0;
	//copy frame into arma structure
	for (u32 i = 0; i < height; i++)
	{
		for (u32 j = 0; j < width; j++)
		{

			tmp_arma_frame(i, j) = (double) in_frame.at<float> (i, j);

		}
	}


	for (u32 i = 0; i <= height - N; i += ovlstep)
	{
		for (u32 j = 0; j <= width - N; j += ovlstep)
		{

			tmp_mtx = sum(sum(tmp_arma_frame.submat(i, j, (i + N - 1), (j + N - 1))));
			cur_edge_hist(y, x)(fidx) = tmp_mtx(0, 0);

			x++;
		}
		x = 0;
		y++;
	}

}

//store the busy/idle counts into a matrix for easy computation
void AnomalyDetection::extract_spatio_temporal_features(const u32 &fs_idx, const u32 &stage)
{

	string file;
	ofstream out_file;

	file.assign("vec.txt");
	out_file.open("output/mtx.txt", ios::app);

	for (u32 x = 0; x < (1 + xmb); ++x)
	{
		for (u32 y = 0; y < (1 + ymb); ++y)
		{

		}
	}
	out_file.close();


}

void AnomalyDetection::test(const cv::Mat &frame, const cv::Mat &grysc_frame)
{

	umat arma_frame((1 + ymb), (1 + xmb));
	ofstream features_file, mv_features_file, pdf_file, edge_pdf;
	features_file.open("output/mtx1.txt", ios::out);
	mv_features_file.open("output/mv.txt", ios::out);
	pdf_file.open("output/pdf.txt", ios::out);
	edge_pdf.open("output/edge_pdf.txt", ios::app);

	test_frame_cnt = test_binary_masks.size();

	if (test_frame_cnt == 1)
	{
		load_ad_model_params();

	}
	loc.clear();

	if ((s32) temporal_window > (s32) test_frame_cnt + 1)
	{
		collect_frames_for_testing(frame, grysc_frame);
		return;
	}
	else
	{
		collect_frames_for_testing(frame, grysc_frame);
	}

	u32 f_idx = testfrm_idx++;
	ad_image.zeros();
	u32 mid_pt = (temporal_window) / 2;

	for (u32 tw_idx = f_idx, z = 0; tw_idx < (f_idx + temporal_window); ++tw_idx, ++z)
	{
		arma_frame = test_binary_masks[tw_idx];
		for (u32 x = 0; x < (1 + xmb); ++x)
		{
			for (u32 y = 0; y < (1 + ymb); ++y)
			{

				//				tmprl_win_feature(y, x)(z) = arma_frame(y, x);

				if (arma_frame(y, x) > 0)
				{
					tmprl_win_feature(y, x)(z) = find_8connected_neighbours(x, y, arma_frame);
					//					tmprl_win_feature(y, x)(z) = arma_frame(y, x);
				}
				else
				{
					tmprl_win_feature(y, x)(z) = 0;
				}

				tmprl_win_edges(y, x).col(z) = fr_edge_hist_tst(y, x)[tw_idx];

				tmprl_avgblk_mv(y, x).col(z) = fr_avgblk_mv_tst(y, x)[tw_idx];

			}
		}
	}

	Col<double> fv(1);
	for (u32 x = 0; x < (1 + xmb); ++x)
	{
		for (u32 y = 0; y < (1 + ymb); ++y)
		{
			compute_feature_vector(tmprl_win_feature(y, x), fv);
			test_feature(y, x) = fv;
			//			cur_edge_hist(y, x) = sum(tmprl_win_edges(y, x), 1);
			//			cout << mid_pt << endl;
			cur_edge_hist(y, x) = tmprl_win_edges(y, x).col(mid_pt);

			cur_avgblk_mv(y, x) = sum(tmprl_avgblk_mv(y, x), 1);

		}
	}

	double probval;

	for (u32 y = 0; y < (1 + ymb); ++y)
	{

		for (u32 x = 0; x < (1 + xmb); ++x)
		{

			bool ad_flag = false;

#if 1
			probval = test_for_speed(cur_avgblk_mv(y, x), x, y);

			if (probval < speed_thr)
			{
				//								double dist = norm(cur_avgblk_mv(y, x), 1);
				//								cout << dist << " " << probval << endl;
				ad_flag = true;
				//								cout << cur_avgblk_mv(y, x) << endl;
			}
			else
			{
#endif
#if 01
#if 1

				probval = test_for_size(test_feature(y, x), x, y);

				if (probval < 1 * speed_thr)
				{
					//					cout << "tv:" << test_feature(y, x) << endl;
					//					cout << "x: " << x << " " << "y: " << y << endl;
					//					sze_pdf(y, x).print_trans("size pdf");
					//					cout << probval << endl;
					ad_flag = true;
				}

#endif
#if 01
				if (ad_flag == true)
				{

					//					probval = test_for_texture(gm_local_model(y, x), cur_edge_hist(y, x));
					probval = test_for_texture(rep_textures(y, x), cur_edge_hist(y, x));
					//										cout << probval << endl;
					//					cout << "................" << endl;
					//					//
					if (x == 20 && y == 10)
					{
						//						cout << "fr: " << test_frame_cnt << " ";
						//						cout << probval << endl;
						//						//											cout << rep_textures(y, x) << endl;
						//											exit(0);

					}
					if (probval > hist_thr)
					{
						//						cout << probval << endl;
						ad_flag = false;

					}
					//					else
					//					{
					//						cout << probval << endl;
					//						cout << "fr no: " << test_frame_cnt << endl;
					//					}

					//					exit(0);

				}
#endif
#endif

			}

#if 0
			u32 xp, yp;
			xp = 15;
			yp = 8;

			if (x == xp && y == yp)
			{
				cout << test_frame_cnt << endl;

				string lpath = "output/tmp/img_";
				char str[512];
				numtostr(test_frame_cnt, str);

				lpath.append(str);
				lpath.append(".png");
				cv::Mat tmpfr = grysc_frame.clone();
				trace_block(tmpfr, xp, yp);
				cv::imwrite(lpath.c_str(), tmpfr);
				//				cout << lpath.c_str() << endl;
				cv::imshow("edge debug", tmpfr);
				cv::waitKey(2);
				edge_pdf << trans(cur_edge_hist(yp, xp)) << endl;

			}
#endif

			if (ad_flag == true)
			{
				if (tmprl_win_feature(y, x)(mid_pt) != 0)
				{
					debug_mb tmp;
					tmp.coords.x = x;
					tmp.coords.y = y;
					tmp.fv = test_feature(y, x);
					tmp.p_value = probval;
					loc.push_back(tmp);

				}

			}

		}


	}



#if 0
	mat size_feature((1 + ymb), (1 + xmb));
	mat tmp_x((1 + ymb), (1 + xmb));
	mat tmp_y((1 + ymb), (1 + xmb));

	size_feature.zeros();
	tmp_x.zeros();
	tmp_y.zeros();
	u32 lcnt, size_t = 6, speed_t = 0.5;

	for (u32 y = 0; y < (1 + ymb); ++y)
	{
		for (u32 x = 0; x < (1 + xmb); ++x)
		{

			//size
			lcnt = 0;
			for (u32 i = 0; i < local_features(y, x).n_rows; ++i)
			{
				features_file << local_features(y, x).row(i)(0) << " ";
				if (local_features(y, x).row(i)(0) > size_t)
				{
					size_feature(y, x) += local_features(y, x).row(i)(0);
					lcnt++;
				}
			}

			if (lcnt > 0)
			{
				size_feature(y, x) /= lcnt;
			}

			features_file << endl;

			for (u32 i = 0; i < local_features(y, x).n_rows; ++i)
			{
				features_file << norm(tmprl_avgblk_mv_tr(y, x).col(i), 1) << " ";
			}

			//motion vectors
			lcnt = 0;
			for (u32 i = 0; i < tmprl_avgblk_mv_tr(y, x).n_cols; ++i)
			{ // L1 norm
				//				mv_features_file << norm(tmprl_avgblk_mv_tr(y, x)(i), 1) << " ";
				mv_features_file << tmprl_avgblk_mv_tr(y, x).col(i)(0) << " ";

				if (tmprl_avgblk_mv_tr(y, x).col(i)(0) > speed_t)
				{
					tmp_x(y, x) += tmprl_avgblk_mv_tr(y, x).col(i)(0);
					lcnt++;
				}

			}

			if (lcnt > 0)
			{
				tmp_x(y, x) /= lcnt;
			}

			mv_features_file << endl;

			lcnt = 0;
			for (u32 i = 0; i < tmprl_avgblk_mv_tr(y, x).n_cols; ++i)
			{
				mv_features_file << tmprl_avgblk_mv_tr(y, x).col(i)(1) << " ";

				if (tmprl_avgblk_mv_tr(y, x).col(i)(1) > speed_t)
				{
					tmp_y(y, x) += tmprl_avgblk_mv_tr(y, x).col(i)(1);
					lcnt++;
				}
			}

			if (lcnt > 0)
			{
				tmp_y(y, x) /= lcnt;
			}

			for (u32 i = 0; i < spd_pdf(y, x).n_elem; ++i)
			{
				pdf_file << spd_pdf(y, x)(i) << " ";
			}
			//			exit(0);

			//edge histogram
			u32 xpick = 4;
			u32 ypick = 9;

			if (x == xpick && y == ypick)
			for (u32 i = 0; i < tmprl_edge_hist_tr(y, x).n_cols; ++i)
			{
				edge_pdf << trans(tmprl_edge_hist_tr(y, x).col(i)) << endl;

			}
			//			for (u32 i = 0; i < tmprl_edge_hist_tr(y, x).n_cols; ++i)
			//			{
			//				edge_pdf << tmprl_edge_hist_tr(y, x).col(i)(1) << " ";
			//			}
			//			for (u32 i = 0; i < tmprl_edge_hist_tr(y, x).n_cols; ++i)
			//			{
			//				edge_pdf << tmprl_edge_hist_tr(y, x).col(i)(2) << " ";
			//			}
			//			for (u32 i = 0; i < tmprl_edge_hist_tr(y, x).n_cols; ++i)
			//			{
			//				edge_pdf << tmprl_edge_hist_tr(y, x).col(i)(3) << " ";
			//			}

			pdf_file << endl << endl;
			mv_features_file << endl << endl << endl;
			features_file << endl << endl << endl;

		}
	}

	features_file << endl;

	cout << "total pts: " << local_features(0, 0).n_elem << endl;

	size_feature.print("avg. size");
	cout << "............" << endl;
	tmp_x.print("avg. mv x");
	cout << "............" << endl;
	tmp_y.print("avg. mv y");
	features_file.close();
	mv_features_file.close();
	pdf_file.close();
	edge_pdf.close();
	exit(0);
#endif

	//averaging the distances
	//	if (test_frame_cnt == 199)
	//	{
	//		training_dist(0) = training_dist(0) / test_frame_cnt;
	//		training_dist(1) = training_dist(1) / test_frame_cnt;
	//	}
	pdf_file.close();
	features_file.close();
	mv_features_file.close();
	edge_pdf.close();
	//	cout << "I am out: " << test_frame_cnt << endl;
	//	exit(0);

}

void AnomalyDetection::collect_frames_for_training(const cv::Mat &bin_frame, const cv::Mat &grysc_frame)
{

	umat arma_frame((1 + ymb), (1 + xmb));

	ofstream features_file;
	features_file.open("output/mtx.txt", ios::app);

	compute_subwindows_features(bin_frame, arma_frame);
	input_binary_masks.push_back(arma_frame);

	estimate_motion(bin_frame, grysc_frame, arma_frame);

	gabor_filter(grysc_frame);
	//	edge_filter(grysc_frame);

	for (u32 f = 0; f < gbr_prms.theta_arr.n_elem; ++f)
	{
		//		cv::imshow("mainWin2", edge_frames(f));
		//		cv::waitKey(200);
		compute_subwindows_edges(edge_frames(f), bin_frame, f);

	}

	double lacc;
	for (u32 x = 0; x < (1 + xmb); ++x)
	{
		for (u32 y = 0; y < (1 + ymb); ++y)
		{

			fr_avgblk_mv_tr(y, x).push_back(cur_avgblk_mv(y, x));

			//						if (accu(cur_edge_hist(y, x)) > 8)
			//						{
			//							cur_edge_hist(y, x) = cur_edge_hist(y, x) / accu(cur_edge_hist(y, x));
			//						}
			//						else
			//						{
			//							cur_edge_hist(y, x).fill(0);
			//						}

			if (cur_edge_hist(y, x).is_finite() != true)
			{
				cout << cur_edge_hist(y, x) << endl;
				exit(0);
			}

			if (1)//(lacc > 1)
			{
				fr_edge_hist_tr(y, x).push_back(cur_edge_hist(y, x));

#if 0
				u32 xp, yp;
				xp = 9;
				yp = 5;
				if (x == xp && y == yp)
				{
					cout << test_frame_cnt << endl;

					string lpath = "output/tmp/img_";
					char str[512];
					numtostr(fr_edge_hist_tr(y, x).size(), str);

					lpath.append(str);
					lpath.append(".png");
					cv::Mat tmpfr = grysc_frame.clone();
					trace_block(tmpfr, xp, yp);
					cv::imwrite(lpath.c_str(), tmpfr);
					cv::imshow("main2", tmpfr);
					cv::waitKey(2);
					features_file << trans(cur_edge_hist(yp, xp)) << endl;
				}
#endif

			}

		}
	}

	cout << "collecting mask no: " << input_binary_masks.size() << endl;
	features_file.close();

}

void AnomalyDetection::collect_frames_for_testing(const cv::Mat &frame, const cv::Mat &grysc_frame)
{

	umat arma_frame((1 + ymb), (1 + xmb));
	compute_subwindows_features(frame, arma_frame);
	test_binary_masks.push_back(arma_frame);
	//	cout << "testing frame no: " << test_binary_masks.size() << endl;

	estimate_motion(frame, grysc_frame, arma_frame);

	gabor_filter(grysc_frame);
	//	edge_filter(grysc_frame);

	for (u32 f = 0; f < gbr_prms.theta_arr.n_elem; ++f)
	{
		compute_subwindows_edges(edge_frames(f), frame, f);
	}

	for (u32 x = 0; x < (1 + xmb); ++x)
	{
		for (u32 y = 0; y < (1 + ymb); ++y)
		{

			fr_avgblk_mv_tst(y, x).push_back(cur_avgblk_mv(y, x));

			//			cur_edge_hist(y, x).print_trans("edge hist");

			//						if (accu(cur_edge_hist(y, x)) > 8)
			//						{
			//							cur_edge_hist(y, x) = cur_edge_hist(y, x) / accu(cur_edge_hist(y, x));
			//						}
			//						else
			//						{
			//							cur_edge_hist(y, x).fill(0);
			//						}

			fr_edge_hist_tst(y, x).push_back(cur_edge_hist(y, x));

		}
	}

#if 0
	u32 x, y;
	x = 2;
	y = 6;

	u32 x1 = x * ovlstep;
	u32 y1 = y * ovlstep;

	cv::Mat blk_img = frame(cv::Rect(x1, y1, (N), (N)));

	cv::imshow("mainWin2", blk_img);
	cv::waitKey(200);

	string filename = "output/tmp/blks/";
	int num = test_binary_masks.size();
	char buffer[256];
	numtostr(num, buffer);
	filename.append(buffer);

	numtostr(arma_frame(y, x), buffer);
	filename.append("_");
	filename.append(buffer);

	filename.append(".png");
	cv::imwrite(filename.c_str(), blk_img);

	//	cout << "fr: " << num <<  " blk_sum: " <<  arma_frame(y, x) << endl;
#endif

}

void AnomalyDetection::trace_block(cv::Mat &frame, const int x, const int y)
{

	//	 FOR DEBUGGING PURPOSE ONLY+++++++++++++++++++++++++++++++++++++++++++++++++++++
	//	cv::Mat temp = frame.clone();
	cv::rectangle(frame, cvPoint(x * ovlstep, y * ovlstep), cvPoint(x * ovlstep + N, y * ovlstep + N), cvScalar(255,
			255, 255), 1, 8, 0);
	//	cv::imshow("mainWin1", temp);
	//	cvWaitKey(10);
}

void AnomalyDetection::trace_block_color(cv::Mat &bgr_img, const int x, const int y)
{

	cv::Mat ycrcb_img = cv::Mat::zeros(cv::Size(bgr_img.cols, bgr_img.rows), CV_8UC3);
	vector<cv::Mat> img_arr;

	//	img_arr.push_back(frame);
	//	img_arr.push_back(frame);
	//	img_arr.push_back(frame);
	//
	//	// create a 3-channel image
	//	cv::merge(img_arr, bgr_img);
	//
	//	//clear the vector
	//	img_arr.clear();

	// convert image BGR to YCrCb
	cvtColor(bgr_img, ycrcb_img, CV_BGR2YCrCb);

	//set the color intensity
	int frac1, frac2, val = 120;
	frac1 = 128 + 0.5 * val;
	frac2 = 128 - 0.1687 * val;
	//	cout << frac1 <<" " << frac2 << endl;

	//split YCrCb image into 3 separate channels
	cv::split(ycrcb_img, img_arr);

	//modify the Cr and Cb channels

	//Cr channel
	cv::rectangle(img_arr[1], cvPoint(x * ovlstep, y * ovlstep), cvPoint(x * ovlstep + N, y * ovlstep + N), cvScalar(
			frac1), CV_FILLED, 8, 0);
	//Cb channel
	cv::rectangle(img_arr[2], cvPoint(x * ovlstep, y * ovlstep), cvPoint(x * ovlstep + N, y * ovlstep + N), cvScalar(
			frac2), CV_FILLED, 8, 0);

	// stitch back the channels into a 3 channel YCrCb image
	cv::merge(img_arr, ycrcb_img);

	// convert image YCrCb to BGR
	cvtColor(ycrcb_img, bgr_img, CV_YCrCb2BGR);

}

void AnomalyDetection::create_dct_table(int N1)
{

	int k = 0;
	double scale_fac_i;

	for (int m = 0; m < N1; m++)
	{

		for (int n = 0; n < N1; n++)
		{

			scale_fac_i = (m == 0) ? sqrt(1.0 / double(N1)) : sqrt(2.0 / double(N1));
			DCT_coeffs(k++) = scale_fac_i * std::cos(double((math::pi() * m) / (2 * N1) * (2 * n + 1)));
		}
	}

}

void AnomalyDetection::display_feature_value(u32 fidx)
{

	cv::Size imsize(width, height);
	Mat<double> armaframe;
	Mat<double> MBframe;
	armaframe.set_size(height, width);
	MBframe.set_size((1 + ymb), (1 + xmb));

	Mat<double> tmp_mtx;
	u32 x = 0, y = 0;

	double lacc, lmax, lconst;

	for (u32 x = 0; x < (1 + xmb); ++x)
	{
		for (u32 y = 0; y < (1 + ymb); ++y)
		{
			MBframe(y, x) = test_feature(y, x)(0);
		}
	}

	lacc = N * N * temporal_window;

	MBframe = MBframe / lacc;

	//    lmax = max(max(MBframe));

	//    lconst = floor(255/lmax)-1;


	MBframe = 255 * MBframe;

	Mat<u8> nMBframe = conv_to<Mat<u8> >::from(MBframe);

	//      cout << nMBframe << endl;
	//
	MBframe.print("vals");

	for (u32 i = 0; i <= height - N; i += ovlstep)
	{
		for (u32 j = 0; j <= width - N; j += ovlstep)
		{
			tmp_mtx = armaframe.submat(i, j, (i + N - 1), (j + N - 1));
			tmp_mtx.fill(MBframe(y, x));
			armaframe.submat(i, j, (i + N - 1), (j + N - 1)) = tmp_mtx;
			x++;
		}
		x = 0;
		y++;
	}

	//	armaframe.print("armafr");

	/*writing the arma frame into opencv matrix*/
	for (u32 i = 0; i < height; i++)
	{
		for (u32 j = 0; j < width; j++)
		{
			estimated_ad_frame.at<arma::u8> (i, j) = (u8) min(armaframe(i, j) * 1, (double) 255);
		}

	}

	cv::imshow("mainWin3", estimated_ad_frame);
	cvWaitKey(20);

}

void AnomalyDetection::compute_feature_vector(const Col<double> &input_vector, Col<double> &out_fv)
{
	//			Col<double> tmp = dct_mtx * tmprl_win_feature(y, x);
	//			tmp = abs(tmp);
	//			fv(1) = sum(tmp.rows(1, temporal_window - 1));


	//	vec filter_out = smoothing_filter % input_vector;
	vec filter_out = input_vector;
	//		out_fv(0) = sum(filter_out);

	u32 mid_pt = (filter_out.n_elem) / 2;
	out_fv(0) = filter_out(mid_pt);

	//	out_fv(1) = 0;

	//	double lacc = 0;
	//	for (u32 idx = 0; idx < filter_out.n_elem - 1; ++idx)
	//	{
	//		lacc += std::abs(filter_out(idx) - filter_out(idx + 1));
	//	}
	//	out_fv(1) = lacc;

	//	if (accu(out_fv) != 0)
	//	{
	//		cout << "raw: " << trans(input_vector) << endl;
	//		cout << "filt: " << trans(filter_out) << endl;
	//		cout << "fv: " << trans(out_fv) << endl;
	//
	//	}

}

void AnomalyDetection::save_ad_model_params()
{
	string rpath = "output/ad_model_prms/ad_model";
	string path;
	string tmp, tmp1;
	char str[128];

	path.assign(rpath);
	path.append("_size");

	//	for (u32 y = 0; y < (1 + ymb); ++y)
	//	{
	//		numtostr(y, str);
	//		tmp.assign("y");
	//		tmp.append(str);
	//
	//		for (u32 x = 0; x < (1 + xmb); ++x)
	//		{
	//			tmp1.assign(path);
	//			tmp1.append(tmp);
	//			numtostr(x, str);
	//			tmp1.append("x");
	//			tmp1.append(str);
	//			local_features(y, x).save(tmp1);
	//			tmp1.clear();
	//
	//		}
	//
	//	}
	path.assign(rpath);
	path.append("_edge");
	tmprl_edge_hist_tr.save(path);

	//	for (u32 y = 0; y < (1 + ymb); ++y)
	//	{
	//		numtostr(y, str);
	//		tmp.assign("y");
	//		tmp.append(str);
	//
	//		for (u32 x = 0; x < (1 + xmb); ++x)
	//		{
	//			tmp1.assign(path);
	//			tmp1.append(tmp);
	//			numtostr(x, str);
	//			tmp1.append("x");
	//			tmp1.append(str);
	//			tmprl_edge_hist_tr(y, x).save(tmp1);
	//			tmp1.clear();
	//
	//		}
	//
	//	}

	path.assign(rpath);
	path.append("_mv");

	tmprl_avgblk_mv_tr.save(path);

	path.assign(rpath);
	path.append("_gmm");
	//	gm_local_model.save(path);


	for (u32 y = 0; y < (1 + ymb); ++y)
	{
		numtostr(y, str);
		tmp.assign("y");
		tmp.append(str);

		for (u32 x = 0; x < (1 + xmb); ++x)
		{
			tmp1.assign(path);
			tmp1.append(tmp);
			numtostr(x, str);
			tmp1.append("x");
			tmp1.append(str);
//			gm_local_model(y, x).save(tmp1);
			tmp1.clear();

		}

	}

	path.assign(rpath);
	path.append("_pdf");
	spd_pdf.save(path);

	path.assign(rpath);
	path.append("_sze_pdf");
	sze_pdf.save(path);

	path.assign(rpath);
	path.append("_rep_txt");
	rep_textures.save(path);

	cout << "ad model saved" << endl;
}

void AnomalyDetection::load_ad_model_params()
{
	string rpath = "output/ad_model_prms/ad_model";
	string path;
	string tmp, tmp1;
	char str[128];

	path.assign(rpath);
	path.append("_size");

	//	for (u32 y = 0; y < (1 + ymb); ++y)
	//	{
	//		numtostr(y, str);
	//		tmp.assign("y");
	//		tmp.append(str);
	//
	//		for (u32 x = 0; x < (1 + xmb); ++x)
	//		{
	//			tmp1.assign(path);
	//			tmp1.append(tmp);
	//			numtostr(x, str);
	//			tmp1.append("x");
	//			tmp1.append(str);
	//			local_features(y, x).load(tmp1);
	//			tmp1.clear();
	//
	//		}
	//
	//	}
	path.assign(rpath);
	path.append("_edge");
	tmprl_edge_hist_tr.load(path);

	//	path.assign(rpath);
	//	path.append("_gmm");
	//	//	gm_local_model.load(path);
	//
	//	for (u32 y = 0; y < (1 + ymb); ++y)
	//	{
	//		numtostr(y, str);
	//		tmp.assign("y");
	//		tmp.append(str);
	//
	//		for (u32 x = 0; x < (1 + xmb); ++x)
	//		{
	//			tmp1.assign(path);
	//			tmp1.append(tmp);
	//			numtostr(x, str);
	//			tmp1.append("x");
	//			tmp1.append(str);
	//			gm_local_model(y, x).load(tmp1);
	//			tmp1.clear();
	//
	//		}
	//
	//	}
	path.assign(rpath);
	path.append("_mv");

	tmprl_avgblk_mv_tr.load(path);

	//	for (u32 y = 0; y < (1 + ymb); ++y)
	//	{
	//		numtostr(y, str);
	//		tmp.assign("y");
	//		tmp.append(str);
	//
	//		for (u32 x = 0; x < (1 + xmb); ++x)
	//		{
	//			tmp1.assign(path);
	//			tmp1.append(tmp);
	//			numtostr(x, str);
	//			tmp1.append("x");
	//			tmp1.append(str);
	//			tmprl_avgblk_mv_tr(y, x).load(tmp1);
	//			tmp1.clear();
	//
	//		}
	//
	//	}

	path.assign(rpath);
	path.append("_pdf");
	spd_pdf.load(path);

	path.assign(rpath);
	path.append("_sze_pdf");
	sze_pdf.load(path);

	path.assign(rpath);
	path.append("_rep_txt");
	rep_textures.load(path);

	cout << "ad model loaded" << endl;

}

void AnomalyDetection::numtostr(int num, char *str)
{
	int i = 0;
	int temp = num;
	char arr[128];

	if (temp == 0)
	{
		str[i] = 0x30;
		str[i + 1] = '\0';
		return;
	}

	while (temp)
	{
		int q = temp % 10;
		temp = temp / 10;
		arr[i] = (unsigned char) ((q) + 0x30);
		i++;
	}

	arr[i] = '\0';
	int len = strlen(arr);
	i = 0;
	for (int var = len - 1; var >= 0; var--)
	{
		str[i] = arr[var];
		i++;
	}
	str[i] = '\0';

}
//
void AnomalyDetection::gabor_filter(const cv::Mat &in_frame)
{

	cv::Mat tmp_img = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	cv::Mat tmp_mtx = cv::Mat::zeros(cv::Size(gbr_prms.kernel_size, gbr_prms.kernel_size), CV_32FC1);
	cv::Mat flt_img = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	in_frame.convertTo(flt_img, flt_img.type());

	for (u32 t = 0; t < gbr_prms.theta_arr.n_elem; ++t)
	{


		cv::flip(gbr_prms.kernel_mtx(t), tmp_mtx, -1);
		cv::filter2D(flt_img, tmp_img, tmp_img.depth(), tmp_mtx);
		edge_frames(t) = cv::abs(tmp_img);

	}

}

void AnomalyDetection::gabor_kernel_initialisation()
{
	s32 x, y;
	float kernel_val;
	ofstream gabor_file;
	gabor_file.open("output/gabor.txt", ios::out);
	cv::Mat tmp_img = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

	/*******************************************create mask****************************/

	cv::Mat mask = cv::Mat::ones(cv::Size(gbr_prms.kernel_size, gbr_prms.kernel_size), CV_32FC1);
	u32 c_coord = ceil(gbr_prms.kernel_size / 2);
	float dist, size_thr = 4;

	for (x = 0; x < gbr_prms.kernel_size; x++)
	{
		for (y = 0; y < gbr_prms.kernel_size; y++)
		{
			dist = sqrt((x - c_coord) * (x - c_coord) + (y - c_coord) * (y - c_coord));

			if (dist > size_thr)
			{
				mask.at<float> (y, x) = 0;
			}
		}
	}

	//	cout << mask << endl;
	//	exit(0);

	/******************************************* create mask****************************/

	float x_p, y_p, x_2, y_2;
	float w0 = (2 * math::pi()) / gbr_prms.lambda;

	//frequency bandwidth of 1.5 octave
	float k = math::pi(); // 2.5
	float phi = exp(-(k * k) / 2);

	for (u32 t = 0; t < gbr_prms.theta_arr.n_elem; ++t)
	{

		float theta = CV_PI * gbr_prms.theta_arr(t) / 180;

		//				cout << "theta :" << gbr_prms.theta_arr(t) << endl;

		s32 ofset = gbr_prms.kernel_size / 2;
		float norm_factor = (w0 / (sqrt(2 * math::pi()) * k));
		float exp_val, cosine_part;

		for (x = 0; x < gbr_prms.kernel_size; x++)
		{
			for (y = 0; y < gbr_prms.kernel_size; y++)
			{
				x_2 = (float) (x - ofset);
				y_2 = (float) (y - ofset);

				//				cout << x_2 << " " << y_2 << endl;

				x_p = ((x_2 * cos(theta)) + (y_2 * sin(theta)));
				y_p = ((-x_2 * sin(theta)) + (y_2 * cos(theta)));

				exp_val = ((w0 * w0) / (8 * k * k)) * (4 * x_p * x_p + y_p * y_p);
				cosine_part = cos(w0 * x_p) - phi;

				kernel_val = norm_factor * exp(-exp_val) * cosine_part * mask.at<float> (y, x);

				gbr_prms.kernel_mtx(t).at<float> (x, y) = kernel_val;

				//								cout << kernel_val << "  ";
				gabor_file << kernel_val << "  ";
			}

			//						cout << endl;
			gabor_file << endl;
		}

		//		cout << endl << endl << endl;
		gabor_file << endl << endl << endl;

	}
	gabor_file.close();

	//					exit(-1);


}

void AnomalyDetection::edge_filter(const cv::Mat &in_frame)
{

	cv::Mat tmp_img = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

	ofstream edge_file;
	edge_file.open("output/edge_out.txt", ios::out);
	//	cv::Mat flag =  cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	//	cv::Mat dst =  cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

	for (u32 t = 0; t < edgefilt_prms.n_kernels; ++t)
	{
		cv::filter2D(in_frame, tmp_img, CV_32F, edgefilt_prms.kernel_mtx(t));

		edge_frames(t) = cv::abs(tmp_img);

		//		cv::multiply(edge_frames(t), flag, dst);
		//		edge_frames(t) = tmp_img;
		//		cout <<"t: "<< t << " "  << edgefilt_prms.kernel_mtx(t) << endl;
		//		edge_file << endl << endl;

		//		const cv::Mat tmp = edge_frames(t);
		//				cv::Scalar lsum = cv::sum(tmp);
		//				cout << "sum: "  <<lsum[0] << " " << lsum[1] << endl;
						cv::imshow("mainWin2", tmp_img);
						cv::waitKey(500);
	}

	//	dst = cv::abs(edge_frames(0)) + cv::abs(edge_frames(1));

	//	cv::compare(edge_frames(t), 20.0, flag, cv::CMP_GT);

	edge_file.close();

	//		exit(0);
}

void AnomalyDetection::edge_filter_initialisation()
{

	mat A;

	double scale = 1.0 / 128;
	/********************************orientation = 0 ******************************************/

	A << -1 << -2 << -1 << endr << 0 << 0 << 0 << endr << 1 << 2 << 1 << endr;

	//	A   << 1 << 2 << 0 << -2 << -1 << endr
	//		<< 4 << 8 << 0 << -8 << -4 << endr
	//		<< 6 << 12 << 0 << -12 << -6 << endr
	//		<< 4 << 8 << 0 << -8 << -4 << endr
	//		<< 1 << 2 << 0 << -2 << -1 << endr;


	A = A * scale;

	for (s32 x = 0; x < edgefilt_prms.kernel_size; x++)
	{
		for (s32 y = 0; y < edgefilt_prms.kernel_size; y++)
		{
			edgefilt_prms.kernel_mtx(0).at<float> (y, x) = A(y, x);
		}
	}

	/********************************orientation = 45 ******************************************/

	A << -2 << -1 << 0 << endr << -1 << 0 << 1 << endr << 0 << 1 << 2 << endr;

	A = A * scale;

	for (s32 x = 0; x < edgefilt_prms.kernel_size; x++)
	{
		for (s32 y = 0; y < edgefilt_prms.kernel_size; y++)
		{
			edgefilt_prms.kernel_mtx(1).at<float> (y, x) = A(y, x);
		}
	}
	/********************************orientation = 90 ******************************************/

	A << -1 << 0 << 1 << endr << -2 << 0 << 2 << endr << -1 << 0 << 1 << endr;

	A = A * scale;
	for (s32 x = 0; x < edgefilt_prms.kernel_size; x++)
	{
		for (s32 y = 0; y < edgefilt_prms.kernel_size; y++)
		{
			edgefilt_prms.kernel_mtx(2).at<float> (y, x) = A(y, x);
		}
	}

	/********************************orientation = 135 ******************************************/

	A << 0 << 1 << 2 << endr << -1 << 0 << 1 << endr << -2 << -1 << 0 << endr;

	A = A * scale;
	for (s32 x = 0; x < edgefilt_prms.kernel_size; x++)
	{
		for (s32 y = 0; y < edgefilt_prms.kernel_size; y++)
		{
			edgefilt_prms.kernel_mtx(3).at<float> (y, x) = A(y, x);
		}
	}

	//	cout << edgefilt_prms.kernel_mtx(0) << endl;
	//	cout << edgefilt_prms.kernel_mtx(1) << endl;
	//	cout << "........." << endl;
	//	exit(0);

}

//double anomaly_detection::test_for_texture(const mog_diag<double> &model, const Col<double> &query_vec)
//{
//
//	u32 size = model.n_gaus;
//	u32 lmatch = 0;
//
//	Col<double> dist(size);
//	Col<double> dist1(size);
//
//	dist.zeros();
//	dist1.zeros();
//
//	if (accu(query_vec) == 0)
//	{
//		return (1.0);
//	}
//
//	for (u32 i = 0; i < size; ++i)
//	{
//
//		if (accu(model.means(i)) > 0)
//		{
//
//			//			dist(i) = measure_correlation_coefficient(query_vec, model.means(i));
//			dist(i) = model.log_lhood_single(query_vec, i);
//
//			lmatch++;
//		}
//	}
//
//	double prob = max(dist);
//	return (prob);
//
//}

double AnomalyDetection::test_for_texture(const mat &model, const Col<double> &query_vec)
{

	u32 size = model.n_cols;
	u32 lmatch = 0;

	Col<double> dist(size);

	dist.zeros();

	if (accu(query_vec) == 0)
	{
		return (1.0);
	}

	for (u32 i = 0; i < size; ++i)
	{
		if (accu(model.col(i)) > 0)
		{
			dist(i) = measure_correlation_coefficient(query_vec, model.col(i));
			lmatch++;
		}
	}

	double prob = max(dist);
	//	cout << "p: " << prob << endl;

	return (prob);

}

inline double AnomalyDetection::measure_correlation_coefficient(const Col<double>& a, const Col<double> &b)
{

	double rep_mu, cur_mu, ccv, rr;
	double nmr1, nmr2, nmr_res;

	//	a.print("sample");
	//	b.print("model mu");

	Col<double> sf_cur_vec(a.n_elem);
	Col<double> sf_rep_vec(a.n_elem);

	rep_mu = mean(a);
	cur_mu = mean(b);

	sf_cur_vec = b - cur_mu;
	sf_rep_vec = a - rep_mu;

	nmr1 = sqrt(sum(sum(square(sf_cur_vec))));
	nmr2 = sqrt(sum(sum(square(sf_rep_vec))));
	nmr_res = nmr1 * nmr2;

	if (nmr_res != 0)
	{
		ccv = dot(sf_cur_vec, sf_rep_vec);
		rr = abs(ccv / nmr_res);

	}
	else
	{
		rr = 1;
	}

	return rr;

}

#if 0

double AnomalyDetection::test_for_speed(const field<Col<double> > &data,
		const Col<double> &query_vec, const double &bw, double &out_avg_dist)
{

#define FILE_OP
#ifdef FILE_OP
	ofstream features_file;
	features_file.open("output/speed_features.txt", ios::out|ios::app);
#endif

	u32 size = data.n_elem;
	Col<double> dist(size);
	u32 n_samples = 0, lmatch = 0;
	double prob;
	vec tmp(2);
	out_avg_dist = 0;
	prob = 0;

	if (sum(query_vec) < 0.5)
	{
		return (0.9);
	}

	for (u32 i = 0; i < size; ++i)
	{

		if (sum(data(i)) > 0)
		{
			tmp = ((query_vec - data(i)));

			double lnorm = norm(tmp, 2);
			double lnorm1 = norm(tmp, 1);
#ifdef FILE_OP
			features_file << tmp(0) << " " << tmp(1) << endl;
			//			features_file << lnorm1 << " " << lnorm << endl;
#endif
			uvec flg = tmp < bw;

			if (sum(flg) == flg.n_elem)
			{
				lmatch++;
			}

			n_samples++;

		}
	}

	if (n_samples > 0)
	{
		prob = double(lmatch) / (double) n_samples;

	}

	return (prob);
#ifdef FILE_OP
	features_file.close();
#endif

}

double AnomalyDetection::test_for_size(const field<Col<double> > &data,
		const Col<double> &query_vec, const double &bw, double &out_avg_dist)
{

	u32 size = data.n_elem;
	u32 n_samples = 0, lmatch = 0;
	Col<double> dist(size);
	double prob, tmp;
	out_avg_dist = 0;
	prob = 0;

	if ((query_vec(0)) < ofset)
	{
		return (0.9);
	}

	for (u32 i = 0; i < size; ++i)
	{

		if (data(i)(0) > ofset)
		{

			tmp = query_vec(0) - data(i)(0);

			if (tmp < bw)
			{
				lmatch++;
			}
			n_samples++;
		}

	}

	if (n_samples > 0)
	{
		prob = (double) lmatch / (double) n_samples;
	}

	return (prob);

}
#endif

double AnomalyDetection::test_for_speed(const Col<double> &query_vec, const u32 &x, const u32 &y)
{

	double dist = norm(query_vec, 1);
	u32 i;
	s32 match_idx = 0;
	double probval = 1;
	if (dist > 0.5)
	{

		if (dist < spd_kd_eval_pts(0))
		{
			match_idx = 0;
		}
		else if (dist > spd_kd_eval_pts(spd_kd_eval_pts.n_elem - 1))
		{
			match_idx = -1;
		}
		else
		{
			for (i = 1; i < spd_kd_eval_pts.n_elem; ++i)
			{
				if (spd_kd_eval_pts(i) > dist)
				{
					match_idx = i;
					break;
				}
			}
		}

		if (match_idx == -1)
		{
			probval = 0;
		}
		else if (match_idx == 0)
		{
			probval = spd_pdf(y, x)(match_idx);
		}
		else
		{
			probval = (spd_pdf(y, x)(match_idx) + spd_pdf(y, x)(match_idx - 1)) / 2;
		}

		//		if (probval < speed_thr)
		//		{
		//			cout << "match_idx " << match_idx << " probval: " << probval << endl;
		//			spd_pdf(y, x).print_trans("pdf");
		//			cout << " ..............." << endl;
		//		}

	}

	return (probval);

}

double AnomalyDetection::test_for_size(const Col<double> &query_vec, const u32 &x, const u32 &y)
{

	double dist = query_vec(0);
	u32 i;
	s32 match_idx = 0;
	double probval = 1;
	if (dist > 5)
	{

		if (dist < sze_kd_eval_pts(0))
		{
			match_idx = 0;
		}
		else if (dist > sze_kd_eval_pts(sze_kd_eval_pts.n_elem - 1))
		{
			match_idx = -1;
		}
		else
		{
			for (i = 1; i < sze_kd_eval_pts.n_elem; ++i)
			{
				if (sze_kd_eval_pts(i) > dist)
				{
					match_idx = i;
					break;
				}
			}
		}

		if (match_idx == -1)
		{
			probval = 0;
		}
		else if (match_idx == 0)
		{
			probval = sze_pdf(y, x)(match_idx);
		}
		else
		{
			probval = (sze_pdf(y, x)(match_idx) + sze_pdf(y, x)(match_idx - 1)) / 2;
		}
	}

	return (probval);

}

void AnomalyDetection::estimate_motion(const cv::Mat &bin_frame, const cv::Mat &grysc_frame, const umat &sub_win_mtx)
{

	cv::VideoCapture cap;
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	cv::Size winSize(15, 15);

	cur_frame = grysc_frame.clone();

	vector<cv::Point2f> points[2];

	cv::Point2f pt;

	//copy only active pixel locations
	for (u32 i = 0; i < height; i += 1)
	{
		for (u32 j = 0; j < width; j += 1)
		{

			if ((bin_frame.at<arma::u8> (i, j) & 0x1) == 1)
			{

				pt.x = j;
				pt.y = i;
				points[0].push_back(pt);

			}

		}
	}
	//reset motion vectors to zero
	mv_mtx_x.zeros();
	mv_mtx_y.zeros();

	vector<uchar> status;
	vector<float> err;

	umat error;
	error.set_size(height, width);
	error.zeros();

	// copy first frame only
	if (prev_frame.empty())
	{
		cur_frame.copyTo(prev_frame);
	}

	calcOpticalFlowPyrLK(prev_frame, cur_frame, points[0], points[1], status, err, winSize, 0, termcrit);

	if ((points[0].size() != points[1].size()) && (err.size() != status.size()) && (err.size() != points[1].size()))

	{
		cout << "mismatch" << endl;
		exit(0);
	}

	//	cout << "total points: " << points[1].size() << endl;


	//	cv::Mat image;
	//	image = grysc_frame.clone();
	//	cv::namedWindow("LK Demo", 1);
	//	float diff;

	u32 k = 0, z = 0;
	for (u32 i = 0; i < height; i += 1)
	{
		for (u32 j = 0; j < width; j += 1)
		{

			if ((bin_frame.at<arma::u8> (i, j) & 0x1) == 1)
			{

				if (status[z])
				{
					mv_mtx_x(i, j) = std::abs(points[1][z].x - points[0][z].x);
					mv_mtx_y(i, j) = std::abs(points[1][z].y - points[0][z].y);

					//					error(i, j) = 1;

					//					diff = norm(points[0][z] - points[1][z]);
					//
					//					if (diff > 3)
					//					{
					//						//						cout << "diff: " << diff << endl;
					//					cout << "x: " << mv_mtx_x(i, j) << " y: "
					//							<< mv_mtx_y(i, j) << endl;
					//						//cout << points[0][z].x << " " << points[0][z].y  << endl;
					//
					//						points[1][k++] = points[1][z];
					//						cv::circle(image, points[1][z], 3, cv::Scalar(255), -1,
					//								8);
					//					}
				}

				z++;
			}

		}
	}
	u32 x = 0, y = 0;
	mat tmp_mtx(1, 1);
	//analyse region and take a call to classify it as idle/busy block
	for (u32 i = 0; i <= height - N; i += ovlstep)
	{
		for (u32 j = 0; j <= width - N; j += ovlstep)
		{
			cur_avgblk_mv(y, x).zeros();

			if (sub_win_mtx(y, x) > 0)
			{

				//				cout << mv_mtx_x.submat(i, j, (i + N - 1), (j + N - 1)) << endl;

				tmp_mtx = sum(sum(mv_mtx_x.submat(i, j, (i + N - 1), (j + N - 1))));
				cur_avgblk_mv(y, x)(0) = (tmp_mtx(0, 0)) / (sub_win_mtx(y, x));

				tmp_mtx = sum(sum(mv_mtx_y.submat(i, j, (i + N - 1), (j + N - 1))));
				cur_avgblk_mv(y, x)(1) = (tmp_mtx(0, 0)) / (sub_win_mtx(y, x));

				//				if (norm(cur_avgblk_mv(y, x), 1) > 50)
				//				{
				//					mat tmp = mv_mtx_x.submat(i, j, (i + N - 1), (j + N - 1));
				//					tmp.print("error x ");
				//					tmp = mv_mtx_y.submat(i, j, (i + N - 1), (j + N - 1));
				//					tmp.print("error y");
				//					cout << "i: " << i << " j: " << j << endl;
				//					umat tmp1 = error.submat(i, j, (i + N - 1), (j + N - 1));
				//					tmp1.print("error st");
				//
				//					cout << sub_win_mtx(y, x)  << endl;
				//					cv::waitKey(500);
				//				}

				//				cout << cur_avgblk_mv(y, x)(0) << " " << cur_avgblk_mv(y, x)(1) << endl;
			}
			x++;
		}
		x = 0;
		y++;
	}

	//if(test_frame_cnt > 1)
	//{
	//	exit(0);
	//}

	//	for (u32 x = 0; x < (1 + xmb); ++x)
	//	{
	//		for (u32 y = 0; y < (1 + ymb); ++y)
	//		{
	//			if (norm(cur_avgblk_mv(y, x), 1) > 50)
	//			{
	//				cout << norm(cur_avgblk_mv(y, x), 1) << endl;
	//				u32 i = x * N;
	//				u32 j = y * N;
	//
	//			}
	//		}
	//	}

	//	}

	//	cv::imshow("LK Demo", gray);
	//	cv::waitKey(-1);


	//	cv::imshow("LK Demo", image);
	//	cv::waitKey(2);
	//	std::swap(points[1], points[0]);

	cv::swap(prev_frame, cur_frame);

}

double AnomalyDetection::find_8connected_neighbours(const u32 &x, const u32 &y, const umat &blk_occ_mtx)
{

	int tr, lc, rc, br;
	int top, left, right, bottom;
	(y > 0) ? top = 1 : top = 0;
	(y < ymb) ? bottom = 1 : bottom = 0;

	(x > 0) ? left = 1 : left = 0;
	(x < xmb) ? right = 1 : right = 0;

	(top == 0) ? tr = 1 : tr = 0;
	(bottom == 0) ? br = 1 : br = 0;
	(right == 0) ? rc = 1 : rc = 0;
	(left == 0) ? lc = 1 : lc = 0;

	umat sub_mtx = blk_occ_mtx.submat((y - 1 + tr), (x - 1 + lc), (y + 1 - br), (x + 1 - rc));
	double mean_val;

	if (sub_mtx.n_elem == 9)
	{
		mean_val = accu(sub_mtx % smoothing_filter);

		//				cout << sub_mtx % smoothing_filter << endl;
	}
	else
	{
		mean_val = (double) accu(sub_mtx) / sub_mtx.n_elem;
	}

	//	mat filt(sub_mtx.n_rows, sub_mtx.n_cols);
	//    mat out = sub_mtx % filt;


	//	cout << "x: " << x << " y: " << y << endl;
	//	sub_mtx.print("sub mtx");
	//	cout << "mean: " << mean_val << " " << blk_occ_mtx(y, x) << endl;
	//	cout << ".............." << endl;


	return mean_val;

}

void AnomalyDetection::kernel_density_estimate(const mat &training_samples, vec &blk_mv_pdf, const u32 &idx)
{
#if 1
	double kd_bw[2] =
	{ 0.25, 3 };
	double kd_bw_inv;
	kd_bw_inv = 1 / kd_bw[idx];

	mat eval_pts = linspace<mat> (kd_min[idx], kd_max[idx], n_kd_eval_pts[idx] + 1);

	//	training_samples.print("tr");
	mat data = training_samples * kd_bw_inv;
	eval_pts = eval_pts * kd_bw_inv;

	u32 n_eval_pts = eval_pts.n_rows;
	u32 n_data_pts = data.n_rows;

	mat W(n_eval_pts, n_data_pts);

	double norm_const = (1 / sqrt(2 * math::pi()));

	for (u32 i = 0; i < n_eval_pts; ++i)
	{
		mat z = data - repmat(eval_pts.row(i), n_data_pts, 1);

		z = norm_const * exp(-square(z) / 2);
		W.row(i) = trans(z);

		//		data.print("x");
		//		eval_pts.print("evlpts");
		//		z.print("diff");
		//		exit(-1);
	}

	blk_mv_pdf = sum(W, 1) / n_data_pts;

	blk_mv_pdf = blk_mv_pdf / accu(blk_mv_pdf);

	//	eval_pts.print("evlpts");
	//		blk_mv_pdf.print("pdf");
	//
	//		exit(0);
	//
	//	training_samples.print_trans("data");
	//	blk_mv_pdf.print_trans("pdf");

#else

	//	mat training_samples1;
	//	training_samples1 << 0.852323 << endr << 0.820921 << endr << 0.646209 << endr << 0.990593 << endr << 0.554856
	//			<< endr << 0.143459 << endr << 0.497505 << endr << 0.068439 << endr << 0.965635 << endr << 0.587238 << endr
	//			<< 0.318265 << endr << 0.228969 << endr << 0.168452 << endr << 0.778257 << endr << 0.338582 << endr
	//			<< 0.790529 << endr << 0.423388 << endr << 0.411963 << endr << 0.048688 << endr << 0.448737 << endr;

	//	training_samples1 = training_samples1 * 1;


	mat eval_pts = linspace<mat> (kd_min[idx], kd_max[idx], n_kd_eval_pts[idx] + 1);
	u32 n_eval_pts = eval_pts.n_rows;

	vec c1 = eval_pts.submat(0, 0, n_eval_pts - 2, 0);
	vec c2 = eval_pts.submat(1, 0, n_eval_pts - 1, 0);
	vec cutoff = (c1 + c2) / 2;

	//	cutoff.print_trans("cutoff");
	//	c1.print_trans("c1");
	//	c2.print_trans("c2");
	//	exit(0);


	umat chist = zeros<umat> (n_eval_pts + 1, training_samples.n_cols);

	for (u32 i = 0; i < (n_eval_pts - 1); ++i)
	{
		chist.row(i + 1) = sum(training_samples <= cutoff(i));
	}

	//setting last bin to the size of the dataset
	chist.row(n_eval_pts) = sum(training_samples <= cutoff(n_eval_pts - 2));

	mat hist = conv_to<mat>::from(chist);
	hist.print_trans("hist");
	rowvec acc(1);
	for (u32 i = 0; i < n_eval_pts; ++i)
	{
		acc = hist.row(i + 1) - hist.row(i);
		blk_mv_pdf(i) = sum(acc);
	}
	//histogram normalisation
	blk_mv_pdf = blk_mv_pdf / accu(blk_mv_pdf);

	if (idx == 0)
	{
		training_samples.print_trans("data");
		blk_mv_pdf.print_trans("pdf");
		cout << "..........." << endl;
		//		exit(0);
	}

#endif
}

void AnomalyDetection::online_texture_quantisation(const mat &texture_mat, mat &rep_textures)
{

	bool updateflg, addblkflg;
	double rr;
	vector<running_stat_vec<double> > vq_elem;
	running_stat_vec<double> curr_elem;
	vector<u32> wt_arr;
	u32 wt = 1;
	//	cout << "txt: in " << endl;
	u32 st_no = 0;
	while (st_no < texture_mat.n_cols)
	{
		if (accu(texture_mat.col(st_no)) > 0)
		{
			curr_elem(texture_mat.col(st_no));
			vq_elem.push_back(curr_elem);
			wt_arr.push_back(wt);
			break;
		}
		st_no++;
	}

	if (st_no == texture_mat.n_cols)
	{
		cout << "n : " << st_no << endl;
		rep_textures.set_size(gbr_prms.theta_arr.n_elem, 1);
		rep_textures.col(0) = texture_mat.col(0);
		return;
	}

	st_no++;

	for (u32 i = st_no; i < texture_mat.n_cols; ++i)
	{
		addblkflg = true;
		updateflg = false;

		if (accu(texture_mat.col(i)) > 0)
		{

			for (u32 k = 0; k < vq_elem.size(); ++k)
			{

				rr = measure_correlation_coefficient(vq_elem[k].mean(), texture_mat.col(i));

				if (rr > 0.8)
				{
					updateflg = true;
				}

				//				cout << "corrcoef: " << rr << "   " << vq_elem.size() << endl;

				if (updateflg == true)
				{

					addblkflg = false;
					//					cout << trans(vq_elem[k].mean()) << endl;
					vq_elem[k](texture_mat.col(i));
					wt_arr[k]++;
					//					cout << trans(texture_mat.col(i)) << endl;
					//					cout << trans(vq_elem[k].mean()) << endl;
					//					cout << "..........." << endl;
					break;

				}

			}

			if (addblkflg == true)
			{
				curr_elem.reset();
				curr_elem(texture_mat.col(i));
				vq_elem.push_back(curr_elem);
				wt_arr.push_back(wt);
			}
		}
	}

	rep_textures.set_size(gbr_prms.theta_arr.n_elem, vq_elem.size());
	for (u32 i = 0; i < vq_elem.size(); ++i)
	{
		rep_textures.col(i) = vq_elem[i].mean();
		//		cout << trans(vq_elem[i].mean()) << " ";
		//		cout << wt_arr[i] << endl;
	}

	//	cout << " ..............." << endl;
	//	cout << "txt: out " << endl;
}
