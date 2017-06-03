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


#ifndef SEQUENTIAL_BGE_PARAMS_H_
#define SEQUENTIAL_BGE_PARAMS_H_

#include "inc.h"

typedef struct
{
	double N;
	double Ovlstep;
	double Pixel_Diff;
	double Corr_Coef;
	double Eta;
	double MinFrames;
	double Iterations;

} bg_est_param_values;




//default_TvalsN = 8;
//default_Tvals.Ovlstep = 8;
//default_Tvals.Pixel_Diff = 4;
//default_Tvals.Corr_Coef = 0.8;
//default_Tvals.Eta = 3;
//default_Tvals.MinFrames = 100;
//default_Tvals.Iterations= 10;

class bge_params
{
public:
	const s32 len;
	const u16 N;
	const s16 ovlstep;
	const double  pixel_diff;
	const double  corr_coef;
	const double eta;
	const u32 min_frames;
	const u32 iterations;

	bge_params(const s32 &in_len, const bg_est_param_values &in_Tvals);

	virtual ~bge_params();
};

#endif /* SEQUENTIAL_BGE_PARAMS_H_ */
