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

#ifndef CASCADEDBGSPARAMS_H_
#define CASCADEDBGSPARAMS_H_

#include "inc.h"

typedef struct
{
	 double T1;
	 double rho_fixed;
	 double alpha;
	 double cosinedist_T;
	 double likelihood_ratio_T;
	 double tmprl_cosinedist_T;
	 double use_spatial_coherence_based_bg_est;
	 u32 fv_type;
	 bool one_time_training;
	 bool training_required;
} bgscascade_thresholds;



class CascadedBgsParams
{
public:
	const s32 len;
	const s16 N;
	const s16 sub_mat_elem;
	const s16 ovlstep;
	const u32 n_gaus;
	const u32 n_gaus_final;
	const u32 n_iter;
	const double trust;
	const bool normalise;
	const bool print_progress;
	const	double T1;
	const double rho_fixed;
	const double alpha;
	const double cosinedist_T;
	const double likelihood_ratio_T;
	const double tmprl_cosinedist_T;
	const bool use_spatial_coherence_based_bg_est;
	const u32 fv_type;
	bool one_time_training;
	bool training_required;

	CascadedBgsParams(const s32 in_len,const s16 in_N,const s16 in_fvno,const s16 in_ovlstep,const u32 n_gaus,
			const u32 n_gaus_final,const u32 n_iter,
			const double trust, const bool normalise,
			const bool print_progress,  const bgscascade_thresholds  &T_vals);

	virtual ~CascadedBgsParams();
};


#endif /* CASCADEDBGSPARAMS_H_ */
