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
#include "k_nearest_neighbours_proto.hpp"

//template<typename eT> k_nearest_neighbours<eT>::k_nearest_neighbours()
//{
//	// TODO Auto-generated constructor stub
//
//}

template<typename eT> k_nearest_neighbours<eT>::~k_nearest_neighbours()
{
	// TODO Auto-generated destructor stub
}

template<typename eT> double k_nearest_neighbours<eT>::find_knn_distance(field<
		Col<eT> > data, Col<eT> query_vec)
{

	u32 size = data.n_elem;
	Col<eT> dist(size);

	for (u32 i = 0; i < size; ++i)
	{
		Col<eT> tmp = (abs(data(i) - query_vec));
		dist(i) = sum(tmp);
	}

	vec min_dist = sort(dist);

	//	min_dist.print("min distance array");
	return (min_dist(K - 1));

}

template<typename eT> double k_nearest_neighbours<eT>::find_knn_distance(
		const field<Col<eT> > &data, const Col<eT> &query_vec,
		const Col<double> &bandwidth_array)
{

	u32 size = data.n_elem;
	u32 lmatch = 0;
	Col<eT> dist(size);

	if (accu(query_vec) < 5)
	{
		return (1.0);
	}

	for (u32 i = 0; i < size; ++i)
	{
		Col<eT> tmp = ((query_vec - data(i)));

		u32 dimension_match = 0;
		for (u32 idx = 0; idx < bandwidth_array.n_elem; ++idx)
		{
			if (tmp(idx) < bandwidth_array(idx))
			{
				dimension_match++;
			}

		}
		if (dimension_match == bandwidth_array.n_elem)
		{
			lmatch++;
		}

	}

	double prob = (double) lmatch / (double) size;

	return (prob);

}

template<typename eT> double k_nearest_neighbours<eT>::find_histogram_dist(
		const field<Col<eT> > &data, const Col<eT> &query_vec,
		const Col<eT> &bandwidth_array)
{

	u32 size = data.n_elem;
	u32 lmatch = 0;
	Col<eT> dist(size);
	double cosdist;

	cout << size << endl;

	if ((accu(query_vec) < 1) || (size < 1))
	{
		return (-1.0);
	}

	for (u32 i = 0; i < size; ++i)
	{

		cosdist = 1 - norm_dot(query_vec, data(i));

		//		query_vec.print_trans("x");
		//		data[i].print_trans("data");
		//		cout << cosdist << endl;

		if (cosdist > bandwidth_array(0))
		{
			lmatch++;
		}

	}

	double prob = (double) lmatch / (double) size;

	cout << lmatch << " " << size << "  :" << prob << endl;

	return (prob);

}

template<typename eT> double k_nearest_neighbours<eT>::test_for_speed(
		const field<Col<eT> > &data, const Col<eT> &query_vec,
		const Col<double> &bandwidth_array, double &out_avg_dist)
{

	u32 size = data.n_elem;
	u32 lmatch = 0, n_samples = 0;
	Col<eT> dist(size);
	double prob, tmp;
	out_avg_dist = 0;
	prob = 0;

	if ((query_vec(1)) < 10)
	{
		return (1.0);
	}

	for (u32 i = 0; i < size; ++i)
	{

		if (data(i)(1) > 10)
		{
			tmp = ((query_vec(1) - data(i)(1)));

			//			cout << "d = " << tmp << " x = " << query_vec(1) << " s(i) = " << data(i)(1) << endl;

			if (tmp < bandwidth_array(1))
			{
				lmatch++;
			}

			out_avg_dist += tmp;
			n_samples++;
		}
	}

	if (n_samples > 0)
	{
		prob = (double) lmatch / (double) n_samples;
		out_avg_dist = out_avg_dist / (double) n_samples;
	}

	return (prob);

}

template<typename eT> double k_nearest_neighbours<eT>::test_for_size(
		const field<Col<eT> > &data, const Col<eT> &query_vec,
		const Col<double> &bandwidth_array)
{

	u32 size = data.n_elem;
	u32 lmatch = 0, n_samples = 0;
	Col<eT> dist(size);
	double tmp, prob;

	if ((query_vec(0)) < 16)
	{
		return (1.0);
	}

	for (u32 i = 0; i < size; ++i)
	{

		if (data(i)(1) > 10)
		{

			tmp = query_vec(0) - data(i)(0);

			if (tmp < bandwidth_array(0))
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

template<typename eT> double k_nearest_neighbours<eT>::test_for_texture(
		const field<Col<eT> > &data, const Col<eT> &query_vec,
		const Col<double> &bandwidth_array)
{

	u32 size = data.n_elem;
	u32 lmatch = 0;
	Col<eT> dist(size);
	double score;

	score = 0;
	if (accu(query_vec) < 16)
	{
		return (1.0);
	}

	for (u32 i = 0; i < size; ++i)
	{

		if (accu(data(i)) > 16)
		{
			score += measure_correlation_coefficient(query_vec, data(i));
			lmatch++;
		}
	}

	double prob = score / (double) lmatch;

	return (prob);

}

template<typename eT> inline double k_nearest_neighbours<eT>::measure_correlation_coefficient(
		const Col<eT>& a, const Col<eT> &b)
{

	eT rep_mu, cur_mu, ccv, rr;
	eT nmr1, nmr2, nmr_res;

	Col<eT> sf_cur_vec(a.n_elem);
	Col<eT> sf_rep_vec(a.n_elem);

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
