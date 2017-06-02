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


#ifndef K_NEAREST_NEIGHBOURS_H_
#define K_NEAREST_NEIGHBOURS_H_

template<typename eT>
class k_nearest_neighbours
{

	u32 K;

public:
	k_nearest_neighbours()
	{
		K = 0;
	}
	k_nearest_neighbours(u32 n)
	{
		K = n;
	}

	double find_knn_distance(field<Col<eT> > data, Col<eT> query_vec);

	double find_knn_distance(const field<Col<eT> > &data,
			const Col<eT> &query_vec, const Col<double> &bandwidth_array);

	double test_for_speed(const field<Col<eT> > &data,
			const Col<eT> &query_vec, const Col<double> &bandwidth_array, double &avg_dist);

	double test_for_size(const field<Col<eT> > &data, const Col<eT> &query_vec,
			const Col<double> &bandwidth_array);

	double test_for_texture(const field<Col<eT> > &data, const Col<eT> &query_vec,
			const Col<double> &bandwidth_array);

	inline double measure_correlation_coefficient(const Col<eT>  &a, const Col<eT>  &b);


	double find_histogram_dist(const field<Col<eT> > &data,
			const Col<eT> &query_vec, const Col<eT> &bandwidth_array);

	virtual ~k_nearest_neighbours();
};

#endif /* K_NEAREST_NEIGHBOURS_H_ */
