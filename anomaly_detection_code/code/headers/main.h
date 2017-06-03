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

#ifndef _INCLUDE_MAIN_H
#define _INCLUDE_MAIN_H

#include <sys/types.h>
#include <sys/stat.h>
#include<iomanip>
#include <errno.h>


enum input_type
{
	type_unknown, type_image, type_video
};

#define WRITEMASK
//#define CREATE_INPUT_VIDEO
//#define CREATE_DEMO_VIDEO
#define DS_RATIO  1



#endif
