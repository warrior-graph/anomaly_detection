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

#include "ParameterInfo.h"

ParameterInfo::ParameterInfo(const rowvec &cur_param_vec,
		double down_scale_ratio, bool training_reqd_flag)
{

	//define constant
	acc_secs = 0;
	N = cur_param_vec(0);
	fvno = cur_param_vec(1);
	ovlstep = cur_param_vec(2);
	n_gaus = cur_param_vec(3);
	n_gaus_final = cur_param_vec(4);
	n_iter = cur_param_vec(5);
	total_seqs = cur_param_vec(14);
	ratio = down_scale_ratio;
	in_data_format = type_unknown;
	initialise_algorithm_parameters(cur_param_vec, training_reqd_flag);

	cvNamedWindow("mainWin1", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("mainWin1", 600, 40);
	cvNamedWindow("mainWin2", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("mainWin2", 1000, 40);

}



ParameterInfo::ParameterInfo(const rowvec &cur_param_vec)
{

	//define constant
	acc_secs = 0;
	N = cur_param_vec(1);
	total_seqs = cur_param_vec(0);
	ratio = 1;
	in_data_format = type_unknown;

//	cvNamedWindow("mainWin1", CV_WINDOW_AUTOSIZE);
//	cvMoveWindow("mainWin1", 600, 40);
//	cvNamedWindow("mainWin2", CV_WINDOW_AUTOSIZE);
//	cvMoveWindow("mainWin2", 1000, 40);

}




ParameterInfo::~ParameterInfo()
{
	cout << "exit successfully" << endl;
	double avg_time = 1 / (acc_secs / total_seqs);
	cout << " average time taken per frame on this dataset: " << avg_time
			<< endl;

//	cvDestroyWindow("mainWin1");
//	cvDestroyWindow("mainWin2");

}

void ParameterInfo::initialise_algorithm_parameters(
		const rowvec &cur_param_vec, const bool &training_reqd_flag)
{

	T_vals.T1 = cur_param_vec(6);
	T_vals.rho_fixed = cur_param_vec(7);

	T_vals.alpha = cur_param_vec(8);

	T_vals.cosinedist_T = cur_param_vec(9);
	T_vals.likelihood_ratio_T = cur_param_vec(10);

	T_vals.tmprl_cosinedist_T = cur_param_vec(11);
	T_vals.use_spatial_coherence_based_bg_est = cur_param_vec(12);
	T_vals.fv_type = cur_param_vec(13);
	T_vals.one_time_training = cur_param_vec(15);
	T_vals.training_required = training_reqd_flag;

	T_bge_vals.N = 16;
	T_bge_vals.Ovlstep = 16;
	T_bge_vals.Pixel_Diff = 5;
	T_bge_vals.Corr_Coef = 0.8;
	T_bge_vals.Eta = 3;
	T_bge_vals.MinFrames = 150;
	T_bge_vals.Iterations = 0;

}

void ParameterInfo::downscale_frame_and_pad_if_necessary()
{

	cv::resize(first_img, img_sc, tsize, ratio, ratio);
	true_height = img_sc.rows;
	true_width = img_sc.cols;
	prow = ((img_sc.rows % N != 0)) ? ((img_sc.rows + N - 1) / N) * N
			: img_sc.rows;
	pcol = ((img_sc.cols % N != 0)) ? ((img_sc.cols + N - 1) / N) * N
			: img_sc.cols;

	if (first_img.channels() == 3)
	{
		padded_input_img = cv::Mat(prow, pcol, CV_8UC3, cv::Scalar(0, 0, 0));
	}
	else
	{
		padded_input_img = cv::Mat(prow, pcol, CV_8UC1, cv::Scalar(0));

	}

	height = padded_input_img.rows;
	width = padded_input_img.cols;
	channels = padded_input_img.channels();

}

void ParameterInfo::fill_padded_region()
{

	padded_input_img = cv::Mat(prow, pcol, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::resize(input_img_mtx, img_sc, tsize, ratio, ratio);

	/* fill zeros in padded region of the image */
	cv::copyMakeBorder(img_sc, padded_input_img, 0, (prow - img_sc.rows), 0,
			(pcol - img_sc.cols), cv::BORDER_CONSTANT, 0);
}

void ParameterInfo::upscale_mask(cv::Mat &mask)
{

	cv::Mat act_ds_mask = mask(cv::Rect(0, 0, true_width, true_height));
	cv::resize(act_ds_mask, mask_mtx_us, tsize, (1 / ratio), (1 / ratio),
			CV_INTER_NN);

	roi_image = cv::Mat(cv::Size(pcol,prow), CV_8UC1, cv::Scalar(0));

	for (u32 x = 0; x < pcol; ++x)
	{
		for (u32 y = 0; y < prow; ++y)
		{
			roi_image.at<arma::u8> (y, x) = mask_mtx_us.at<arma::u8> (y, x) &
					padded_input_img.at<arma::u8>(y,x);
		}
	}

}

void ParameterInfo::draw_bounding_boxes()
{

	//Linked list of connected pixel sequences in a binary image
	CvSeq* seq = NULL;
	vector<vector<cv::Point> > contours;
	//Memory allocated for OpenCV function operations
	CvMemStorage* storage = cvCreateMemStorage(0);

	IplImage ipl_img = mask_mtx_us;
	//cv::Mat bin_mask = mask_mtx_us.clone();

	cvFindContours(&ipl_img, storage, &seq, sizeof(CvContour),
			CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));

	cvReleaseMemStorage(&storage);

	//Iterate through segments
	for (; seq; seq = seq->h_next)
	{
		//Find minimal bounding box for each sequence
		CvRect boundbox = cvBoundingRect(seq);
		float area = boundbox.height * boundbox.width;

		if (area > 200)
		{
			cv::Point Pt1, Pt2;
			Pt1.x = boundbox.x;
			Pt1.y = boundbox.y;

			Pt2.x = boundbox.x + boundbox.width;
			Pt2.y = boundbox.y + boundbox.height;

			cv::rectangle(input_img_mtx, Pt1, Pt2, cv::Scalar(0, 0, 255), 2);

		}

	}
	if (seq != NULL)
	{
		cvClearSeq(seq);
	}
}

void ParameterInfo::is_grey_scale_img(const rowvec &cur_param_vec)
{

	if (channels == 1)
	{
		T_vals.T1 = cur_param_vec(6) / 3;
		T_vals.cosinedist_T = cur_param_vec(9) / 3;
		T_vals.tmprl_cosinedist_T = cur_param_vec(11) / 3;
	}

}

void ParameterInfo::load_files_from_folder(
		std::vector<std::string> &out_result)
{

	size_t found;
	DIR *D;
	struct dirent *Dirp;

	//"." Refers to the project location, any path can be placed here
	D = opendir(path.c_str());

	while ((Dirp = readdir(D)) != NULL)
	{
		std::string str(Dirp->d_name);
		found = str.find_last_of(".");
		if (str.substr(found + 1) == "avi" || str.substr(found + 1) == "jpg"
				|| str.substr(found + 1) == "bmp" || str.substr(found + 1)
				== "png" || str.substr(found + 1) == "tif" || str.substr(found
				+ 1) == "mpg")
		{
			out_result.push_back(str);
		}
	}

	closedir(D);
	std::sort(out_result.begin(), out_result.end());
	int length = out_result[0].length();
	if ((out_result[0].compare(length - 3, 3, "jpg") == 0)
			|| (out_result[0].compare(length - 3, 3, "bmp") == 0)
			|| (out_result[0].compare(length - 3, 3, "png") == 0)
			|| (out_result[0].compare(length - 3, 3, "tif") == 0))
	{
		in_data_format = type_image;
	}
	else if ((out_result[0].compare(length - 3, 3, "avi") == 0)
			|| out_result[0].compare(length - 3, 3, "mpg") == 0)
	{
		in_data_format = type_video;
	}

}

void ParameterInfo::get_input_frame(
		const std::vector<std::string> result, u32 frm_idx)
{

	char filename[512];
	string tmp_path;
	if (in_data_format == type_image)
	{
		int length = result[frm_idx].length();
		result[frm_idx].copy(filename, length, 0);
		filename[length] = '\0';
		tmp_path.assign(path);
		tmp_path.append(filename);
		curr_filename.assign(filename);
		input_img_mtx = cv::imread(tmp_path.c_str(), -1);

	}
	else if (in_data_format == type_video)
	{
		vidfile >> input_img_mtx;
	}

	fill_padded_region();

}

void ParameterInfo::get_first_frame(
		const std::vector<std::string> result)
{

	char filename[512];
	string tmp_path;

	int length = result[0].length();
	result[0].copy(filename, length, 0);
	filename[length] = '\0';
	tmp_path.assign(path);
	tmp_path.append(filename);
	cv::Mat tmp_img;
	if (in_data_format == type_image)
	{

		first_img = cv::imread(tmp_path.c_str(), -1);

		if (first_img.empty())
		{
			cout << "Unable to open the input sequence\n";
			exit(-1);
		}
		sequence_len = result.size();
	}
	else if (in_data_format == type_video)
	{
		vidfile.open(tmp_path);
		if (vidfile.isOpened() == false)
		{
			cout << "Unable to open the input video sequence\n";
			exit(-1);
		}

		vidfile >> first_img;
		cout << vidfile.get(CV_CAP_PROP_FRAME_COUNT) << endl;
		sequence_len = vidfile.get(CV_CAP_PROP_FRAME_COUNT);

		cout << "insdie seq len" << sequence_len << endl;

	}
	else
	{
		cout << "unknown input format : not supported" << endl;
		exit(-23);
	}

}

void ParameterInfo::numtostr(int num, char *str)
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

void ParameterInfo::output_statistics()
{

	cout << "Processing a " << width << " x " << height << " image with "
			<< channels << " channels" << endl;
	acc_secs += n_secs / sequence_len;
	cout << "avg time per frame: " << (n_secs / sequence_len) << " seconds"
			<< endl;
	cout << "avg framerate: " << (sequence_len / n_secs) << " fps" << endl;
	cout << endl << "Complete" << endl;
	cout << "+++++++++++++++++++++++++++++++++++++" << endl;
}

void ParameterInfo::load_files_from_filelist(
		std::vector<std::string> &out_result, string filedata)
{

	string str;
	ifstream in_file;
	in_file.open(filedata.c_str(), ifstream::in);

	if (in_file.is_open() == false)
	{
		cout << " Unable to open the input training file list" << endl;
		exit(-23);
	}

	while (!in_file.eof())
	{

		in_file >> str;
		out_result.push_back(str);

	}

	in_data_format = type_image;

	sequence_len = out_result.size();

	in_file.close();
}
