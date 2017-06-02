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
#include "DisplayImage.h"

DisplayImage::DisplayImage(const cv::Size imsize)
{

	disp_image = cv::Mat::zeros(imsize, CV_8UC3);

}

DisplayImage::~DisplayImage()
{

}

void DisplayImage::Overlay_Img(const cv::Mat &img, cv::Size ofset)
{

	int pos_x, pos_y, img_width, img_height;
	pos_x = ofset.width;
	pos_y = ofset.height;

	img_width = img.cols;
	img_height = img.rows;

	if (img_width > disp_image.cols || img_height > disp_image.rows)
	{
		cout << "Overlay image exceeds the given dimensions" << endl;
		cout << " Image Overlaying unsuccessful" << endl;
		return;
	}

int channels = img.channels();

	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{

			if(channels == 3)
			{
			disp_image.at<cv::Vec3b> (pos_y + i, pos_x + j)
					= img.at<cv::Vec3b> (i, j);
			}
			else
			{
				cv::Vec3b pix;
				pix[0] = img.at<arma::u8> (i, j);
				pix[1] = pix[0];
				pix[2] = pix[0];

				disp_image.at<cv::Vec3b> (pos_y + i, pos_x + j)	= pix;
			}

		}

	}

}

void DisplayImage::Overlay_Img(const cv::Mat &img, const cv::Mat &mask,
		cv::Size ofset, const int &colour_fill)
{

	int pos_x, pos_y, img_width, img_height;
	pos_x = ofset.width;
	pos_y = ofset.height;

	img_width = img.cols;
	img_height = img.rows;

	cv::Vec3b pix, colour;


	if (img_width > disp_image.cols || img_height > disp_image.rows)
	{
		cout << "Overlay image exceeds the given dimensions" << endl;
		cout << " Image Overlaying unsuccessful" << endl;
		return;
	}



	switch (colour_fill)
	{
	case green:

		colour[0] = 0;
		colour[1] = 255;
		colour[2] = 0;

		break;
	case blue:

		colour[0] = 255;
		colour[1] = 0;
		colour[2] = 0;

		break;

	case red:
		colour[0] = 0;
		colour[1] = 0;
		colour[2] = 255;
		break;

	default:
		colour[0] = 255;
		colour[1] = 255;
		colour[2] = 255;

		break;

	}

	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{

			if (mask.at<arma::u8> (i, j) == 0)
			{
				pix = img.at<cv::Vec3b> (i, j);
			}
			else
			{
				pix = colour;
			}

			disp_image.at<cv::Vec3b> (pos_y + i, pos_x + j) = pix;

		}

	}

}

cv::Mat DisplayImage::Render_img(void)
{
	return disp_image.clone();
}
