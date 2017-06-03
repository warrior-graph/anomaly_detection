// Please cite the following paper when using this source code:
//
// V. Reddy, C. Sanderson, B.C. Lovell.
// Improved Anomaly Detection in Crowded Scenes via Cell-based Analysis of Foreground Speed, Size and Texture.
// IEEE Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2011.
//
// http://dx.doi.org/10.1109/CVPRW.2011.5981799


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


#include "../headers/inc.h"
#include "../headers/main.h"


#include "../headers/AnomalyDetection.h"
#include "../headers/ParameterInfo.h"

#define  READ_MASKS
#define WRITE_OUTPUT
cv::Mat compute_ad_mask(const AnomalyDetection &obj, const rowvec& in_params);

void numtostr(int num, char *str);

//void Img_mask_concatenate(const cv::Mat &frame, const cv::Mat &bin_img, cv::Mat &out_concatenate_img);
//void compute_cube_mean_std(const field<cube> &global_training_dist);
void ad_post_processing(Mat<uword> &resmtx, cube &ad_flags, const double &twindow, const uword &seqno);
void load_configurations(mat &param_vec);
cv::Mat
filtered_detection(vector<cv::Mat> &display_buffer, AnomalyDetection &obj, const cube & ad_flags, const uword &f);

string outpath = "output/ad_masks/ped1/";



int main(int argc, char **argv)
{
	ios_base::sync_with_stdio(false);
	cout << "using Armadillo " << arma_version::major << "." << arma_version::minor << "." << arma_version::patch
			<< '\n';

	cube ad_flags;
	wall_clock timer;
	mat param_vec;
	vector<cv::Mat> display_buffer;


	ofstream avg_results;

	ofstream features_file;
	features_file.open("output/speed_features.txt", ios::out);
	features_file.close();

	//Setting number of sequences & parameters of the algorithm
	load_configurations(param_vec);

	uword img_width, img_height;
	mat localisation_res;
	string mask_save_path;
	bool ad_training_required = true;
	bool first_time_entry = true;

	for (uword loop_id = 0; loop_id < param_vec.n_rows; ++loop_id)
	{

//		avg_results.open("output/ad_localisation_gist_results.txt", ios::out | ios::app);
//		if (!avg_results)
//		{
//			cout << "Unable to create the avg_results file\n";
//			exit(-1);
//		}

		const rowvec& cur_param_vec = param_vec.row(loop_id);

		cout << cur_param_vec << '\n';

		if (first_time_entry == true)
		{

				img_height = 160;
				img_width = 240;


			uword mbx, mby;

			img_width = ((img_width + cur_param_vec(1) - 1) * cur_param_vec(1)) / cur_param_vec(1);
			img_height = ((img_height + cur_param_vec(1) - 1) * cur_param_vec(1)) / cur_param_vec(1);

			mbx = (img_width - cur_param_vec(1)) / cur_param_vec(2);
			mby = (img_height - cur_param_vec(1)) / cur_param_vec(2);

			mbx++;
			mby++;

			cout << mbx << " " << mby << '\n';


			ad_flags.set_size(mby, mbx, 200);
			ad_flags.zeros();

			first_time_entry = false;



		}

		//create an instance of system configuration to do the front end IO processing
		ParameterInfo file_sys1(cur_param_vec);
		ParameterInfo file_sys2(cur_param_vec);

		//read total sequences
		const u16 total_seqs = cur_param_vec(0);
		Mat<uword> res_mtx;


		string tst_img_path, tr_img_path, seqname, tr_masks_list, tr_imgs_list;

			file_sys1.param_file.open("input/ad_input_fread.txt");




			tr_masks_list = "input/training_masks.txt";
			tr_imgs_list = "input/training_images.txt";

			mask_save_path = "output/ad_masks/ped1/";

			res_mtx.set_size(36, 200);
			res_mtx.fill(0);


		if (!file_sys1.param_file.is_open())
		{
			cout << "Unable to open the input sequence file\n";
			exit(-1);
		}

		for (int seqno = 0; seqno < total_seqs; ++seqno)
		{
			std::vector<std::string> tr_mask_filenames, tr_imgs_filenames, tst_imgs_filenames;
			file_sys1.n_secs = 0;

			file_sys1.param_file >> file_sys1.tr_mask_path >> tr_img_path >> file_sys1.tst_mask_path
					>> tst_img_path >> seqname;
			cout << "Sequence number " << seqno << ": " << seqname << '\n';

#ifdef WRITE_OUTPUT

			string foldername;
			foldername.assign(mask_save_path);
			foldername.append(seqname.c_str());
			if (mkdir(foldername.c_str(), S_IRWXU) == -1)
			{ // Create the directory
				if (errno != EEXIST)
				{
					std::cerr << "Error: " << strerror(errno);
					return (EXIT_FAILURE);
				}
			}

#endif

			//************************************************************************
			cout << "ad train path: " << file_sys1.tr_mask_path << '\n';
			file_sys1.path.assign(file_sys1.tr_mask_path);
			//load files from the folder pointed by member variable 'path'
			//			config_obj.load_files_from_folder(bg_tr_filenames);

			file_sys1.load_files_from_filelist(tr_mask_filenames, tr_masks_list);

			file_sys2.path.assign(tr_img_path);

			file_sys2.load_files_from_filelist(tr_imgs_filenames, tr_imgs_list);

			// load the first image into member variable 'first_img'
			file_sys1.get_first_frame(tr_mask_filenames);
			file_sys2.get_first_frame(tr_imgs_filenames);

			// downsize 'first_img' depending in DS_RATIO and pad to make it multiple of 'N'
			file_sys1.downscale_frame_and_pad_if_necessary();
			file_sys2.downscale_frame_and_pad_if_necessary();

			//create an instance of anomaly detection class
			AnomalyDetection ad_object(file_sys1.padded_input_img, file_sys1.sequence_len, cur_param_vec);

			if (ad_training_required == true)
			{

				for (uword i = 0; i < file_sys1.sequence_len; i++)
				{
					//read frame into 'input_mtx' and then store into 'padded_input_img'
					file_sys1.get_input_frame(tr_mask_filenames, i);
					file_sys2.get_input_frame(tr_imgs_filenames, i);

					//collecting frames for training ad module
					ad_object.collect_frames_for_training(file_sys1.padded_input_img, file_sys2.padded_input_img);

					cv::imshow("mainWin1", file_sys1.input_img_mtx);
					cvWaitKey(2);

				}

				//anomaly detection training
				ad_object.train_model();
				ad_training_required = false;

			}

			cout << "test path: " << file_sys1.tst_mask_path << '\n';

			file_sys1.path.assign(file_sys1.tst_mask_path);
			//load files from the folder pointed by member variable 'path'
			file_sys1.load_files_from_folder(tst_imgs_filenames);
			// load the first image into member variable 'first_img'
			file_sys1.get_first_frame(tst_imgs_filenames);

			std::vector<std::string> ad_tmp_filenames;
			file_sys2.path.assign(tst_img_path);
			file_sys2.load_files_from_folder(ad_tmp_filenames);
			// load the first image into member variable 'first_img'
			file_sys2.get_first_frame(ad_tmp_filenames);
			// downsize 'first_img' depending in DS_RATIO and pad to make it multiple of 'N'
			file_sys2.downscale_frame_and_pad_if_necessary();

			//uword millsec;
			//loop the test images to detect anomaly in the frames
			for (uword f = 0; f < file_sys1.sequence_len; f++)
			{

				file_sys1.get_input_frame(tst_imgs_filenames, f);
				file_sys2.get_input_frame(ad_tmp_filenames, f);

				timer.tic();
				//anomaly detection testing
				ad_object.test(file_sys1.padded_input_img, file_sys2.padded_input_img);
				file_sys1.n_secs += timer.toc();
				//millsec = 2;

				//				cv::Mat ad_mask	= compute_ad_mask(ad_object, cur_param_vec);
#if 0
				cv::Mat ad_mask = file_sys2.padded_input_img.clone();
				ad_object.trace_block(ad_mask, 12, 0);
#endif
				display_buffer.push_back(file_sys2.padded_input_img);

				for (uword i = 0; i < ad_object.loc.size(); i++)
				{

					ad_flags(ad_object.loc[i].coords.y, ad_object.loc[i].coords.x, f) = 1;

					//					cout << f << '\n';
					ad_object.trace_block(file_sys2.input_img_mtx, ad_object.loc[i].coords.x,
							ad_object.loc[i].coords.y);

					ad_object.trace_block(file_sys1.input_img_mtx, ad_object.loc[i].coords.x,
							ad_object.loc[i].coords.y);

				}
				cv::Mat disp_buf;
				if (f > 2)
				{
					disp_buf = filtered_detection(display_buffer, ad_object, ad_flags, f);
				}

#ifdef WRITE_OUTPUT
				if (f > 2)
				{
					string write_path;
					write_path.assign(foldername.c_str());
					write_path.append("/");
					int found;
					if (file_sys1.in_data_format == type_image)
					{
						write_path.append(file_sys1.curr_filename);
						found = write_path.find_last_of(".");
						write_path.replace(found + 1, 3, "png");
					}
					else if (file_sys1.in_data_format == type_video)
					{
						//char lbuff[64];
						string vfile;
						vfile.assign(file_sys1.curr_filename);
						string vid_file = vfile.substr(0, vfile.length() - 4);
						write_path.append(vid_file);
						write_path.append("_");
						// numtostr(f, lbuff);
						write_path.append(to_string(f));
						write_path.append(".png");

					}

					if(disp_buf.channels() != 3)
					{
						cout << " error here " << '\n';
						exit(0);
					}
					//													cout << write_path.c_str() << '\n';
					cv::imwrite(write_path.c_str(), disp_buf);
				}
#endif
#if 0
					cv::imshow("binary", file_sys1.padded_input_img);
					/*************************************************************************/
					//					cv::imshow("mainWin2", config_obj1.input_img_mtx);
					cvWaitKey(millsec);

#endif


			}

			// output the dimensions of the image, frame rate info etc.
			cout << "Seq no: " << seqno << '\n';
			file_sys1.output_statistics();

			ad_post_processing(res_mtx, ad_flags, cur_param_vec(3), seqno);



		}

		file_sys1.param_file.close();

	}

	return 0;
}

void numtostr(int num, char *str)
{
	int i = 0;
	int temp = num;
	char arr[20];

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


cv::Mat compute_ad_mask(const AnomalyDetection &obj, const rowvec& in_params)
{

	uword width, height;

	if (in_params(10) == 0)
	{
		width = 240;
		height = 160;

	}
	else if (in_params(10) == 1)
	{
		width = 360;
		height = 240;
	}
	else
	{
		cout << " compute ad mask: invalid paramter" << '\n';
		exit(-1);
	}

	cv::Size imsize(width, height);
	cv::Mat ad_mask = cv::Mat::zeros(imsize, CV_8UC1);
	uword ovlstep = (uword) in_params(2);
	uword x_cord, y_cord;
	for (uword i = 0; i < obj.loc.size(); i++)
	{
		x_cord = obj.loc[i].coords.x * ovlstep;
		y_cord = obj.loc[i].coords.y * ovlstep;

		for (uword x = x_cord; x < (x_cord + ovlstep); ++x)
		{

			for (uword y = y_cord; y < (y_cord + ovlstep); ++y)
			{

				ad_mask.at<arma::u8> (y, x) = 255;
			}
		}

	}

	uword true_width, true_height;
	if (in_params(10) == 0)
	{
		true_width = 238;
		true_height = 158;
	}
	else
	{
		true_width = 360;
		true_height = 240;
	}

	cv::Mat img_crop(ad_mask, cv::Rect(0, 0, true_width, true_height));

	return (img_crop);

}


void ad_post_processing(Mat<uword> &resmtx, cube &ad_flags, const double &twindow, const uword &seqno)
{

	uword filter_size = 3;
	uword T = 2;
	uword ofset = (filter_size / 2);
	ofstream filt_file;
	filt_file.open("../../wrk_space_output1/filt_gt.txt", ios::out);

	cout << ad_flags.n_rows << " " << ad_flags.n_cols << " " << ad_flags.n_slices << '\n';

	cube ad_flags_raw;
	ad_flags_raw = ad_flags;

	for (uword t = ofset; t < ad_flags.n_slices - ofset; ++t)
	{
		for (uword y = ofset; y < ad_flags.n_rows - ofset; ++y)
		{
			for (uword x = ofset; x < ad_flags.n_cols - ofset; ++x)
			{

				if (ad_flags_raw(y, x, t) > 0)
				{
					cube tmp = ad_flags_raw.subcube(y - ofset, x - ofset, t - ofset, y + ofset, x + ofset, t + ofset);

					if ((accu(tmp.slice(0)) < T) && (accu(tmp.slice(1)) < T) && (accu(tmp.slice(2)) < T))
					{
						ad_flags(y, x, t) = 0;
					}

				}
			}
		}
	}

	uword n_seq = resmtx.n_rows;
	Col<uword> n_frms(n_seq);

	cout << n_seq << '\n';

	if (resmtx.n_rows == 36)
	{
		n_frms.fill(200);
	}
	else if (resmtx.n_rows == 12)
	{
		n_frms.fill(180);
		n_frms(2) = n_frms(4) = n_frms(9) = 150;
		n_frms(8) = 120;
	}
	else
	{
		cout << " invalid no of sequences: ad_post_processing" << '\n';
		exit(0);
	}

	ofset = (uword) (twindow / 2);

	for (uword i = twindow; i < n_frms(seqno); ++i)
	{
		// result affecting the centre frame of the temporal window

		if (accu(ad_flags.slice(i)) > 0)
		{
			resmtx(seqno, i - ofset) = 1;
		}

	}

	cube ad_flags_tmp;
	ad_flags_tmp = ad_flags;
	ad_flags.zeros();

	for (uword t = twindow; t < ad_flags.n_slices; ++t)
	{
		ad_flags.slice(t - ofset) = ad_flags_tmp.slice(t);
	}

	filt_file.close();



}

void load_configurations(mat &param_vec)
{

	Col<uword> prms_srt, prms_end, prms_stp;
	uword dataset_id = 0;


		prms_srt << 16 << 16 << 3 << 40 << 81   << 90 << 6 << 16 << 10;
		prms_end << 16 << 16 << 3 << 40 << 81 << 90 << 6 << 16 << 10;
		prms_stp << 1  << 1  << 1 << 40 << 40  << 1 << 1 << 1 << 1;


	if ((prms_srt.n_elem != prms_end.n_elem) || (prms_srt.n_elem != prms_stp.n_elem) || (prms_stp.n_elem
			!= prms_end.n_elem))
	{
		cout << "load_configurations:incorrect number of parameters" << '\n';
		exit(-1);
	}

	uword size = prms_srt.n_elem;
	float total = 1, tmp;

	for (uword i = 0; i < size; ++i)
	{

		tmp = (uword) (prms_end(i) - prms_srt(i) + prms_stp(i)) / (prms_stp(i));

		cout << tmp << '\n';
		if (tmp < 0)
		{
			cout << "load_configurations:-ve range value" << '\n';
			exit(-1);
		}
		total *= tmp;

	}

	param_vec.set_size((uword) total, size + 2);
	param_vec.zeros();

	// set the number of sequences to be tested here.
	param_vec.col(0).fill(1);
	param_vec.col(param_vec.n_cols - 1).fill(dataset_id);

	uword cnt = 0;
	for (uword i0 = prms_srt(0); i0 <= prms_end(0); i0 += prms_stp(0))
	{
		for (uword i1 = prms_srt(1); i1 <= prms_end(1); i1 += prms_stp(1))
		{
			for (uword i2 = prms_srt(2); i2 <= prms_end(2); i2 += prms_stp(2))
			{
				for (uword i3 = prms_srt(3); i3 <= prms_end(3); i3 += prms_stp(3))
				{
					for (uword i4 = prms_srt(4); i4 <= prms_end(4); i4 += prms_stp(4))
					{
						for (uword i5 = prms_srt(5); i5 <= prms_end(5); i5 += prms_stp(5))
						{

							for (uword i6 = prms_srt(6); i6 <= prms_end(6); i6 += prms_stp(6))
							{
								for (uword i7 = prms_srt(7); i7 <= prms_end(7); i7 += prms_stp(7))
								{

									for (uword i8 = prms_srt(8); i8 <= prms_end(8); i8 += prms_stp(8))
									{
										param_vec(cnt, 1) = i0;
										param_vec(cnt, 2) = i1;
										param_vec(cnt, 3) = i2;
										param_vec(cnt, 4) = (float) i3 / 1000.0;
										param_vec(cnt, 5) = (float) i4 / 10000.0;
										param_vec(cnt, 6) = (float) i5 / 100.0;
										param_vec(cnt, 7) = i6;
										param_vec(cnt, 8) = i7;
										param_vec(cnt, 9) = (float) i8 / 10;

										cnt++;

									}

								}

							}
						}
					}

				}
			}
		}
	}



}
//
cv::Mat filtered_detection(vector<cv::Mat> &display_buffer, AnomalyDetection &obj, const cube & ad_flags, const uword &f)
{

	//vector<cv::Mat>::iterator p = display_buffer.begin();
	ofstream filt_file;
	filt_file.open("../../wrk_space_output1/filt_disp.txt", ios::app);

	uword filter_size = 3;
	uword T = 2;
	uword ofset = (filter_size / 2);

	cube ad_flags_raw;
	ad_flags_raw = ad_flags;

	for (uword y = ofset; y < ad_flags.n_rows - ofset; ++y)
	{
		for (uword x = ofset; x < ad_flags.n_cols - ofset; ++x)
		{

			if (ad_flags_raw(y, x, (f - ofset)) > 0)
			{
				cube tmp = ad_flags_raw.subcube(y - ofset, x - ofset, f - 2 * ofset, y + ofset, x + ofset, f);

				if ((accu(tmp.slice(0)) < T) && (accu(tmp.slice(1)) < T) && (accu(tmp.slice(2)) < T))
				{
					ad_flags_raw(y, x, (f - ofset)) = 0;
				}

			}

		}
	}

	cv::Mat frame = display_buffer[(f - ofset)];
	cv::Mat color_frame = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC3);


	vector<cv::Mat> img_arr;
	img_arr.push_back(frame);
	img_arr.push_back(frame);
	img_arr.push_back(frame);
	// create a 3-channel image
	cv::merge(img_arr, color_frame);
	//clear the vector
	img_arr.clear();


	for (uword y = ofset; y < ad_flags.n_rows - ofset; ++y)
	{
		for (uword x = ofset; x < ad_flags.n_cols - ofset; ++x)
		{

			if (ad_flags_raw(y, x, (f - ofset)) > 0)
			{

				obj.trace_block_color(color_frame, (int) x, (int) y);
			}
		}
	}
	cv::imshow("filtered output", color_frame);
	cv::waitKey(2);

	filt_file.close();

	return color_frame;
}
