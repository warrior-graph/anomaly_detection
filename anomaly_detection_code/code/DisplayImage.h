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


#ifndef DISPLAY_IMAGE_H_
#define DISPLAY_IMAGE_H_

enum colour_fill {red, green, blue, white};

class DisplayImage
{
private:
    cv::Mat disp_image;

public:
	DisplayImage(const cv::Size imsize);
	void Overlay_Img(const cv::Mat &img, const cv::Size ofset);
	void Overlay_Img(const cv::Mat &img, const cv::Mat &mask, cv::Size ofset, const int &colour_fill);
	cv::Mat Render_img(void);

	virtual ~DisplayImage();
};

#endif /* DISPLAY_IMAGE_H_ */
