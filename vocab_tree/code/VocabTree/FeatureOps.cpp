#include <math.h>
#include "FeatureOps.h"

double distanceFromCentre(DescriptorF32 centre, DescriptorUI8 feature)
{
	double distance = 0;

	for(int idx = 0; idx < 128; idx++)
	{
		distance += (centre[idx] - feature[idx]) * (centre[idx] - feature[idx]);
	}

	return distance;
}

void accumulate(DescriptorF32 &acc, DescriptorUI8 feat)
{
	for(int i = 0; i < 128; i++)
	{
		acc[i] = acc[i] + feat[i];
	}
}

void zeroFeat(DescriptorF32 &var)
{
	for(int i = 0; i < 128; i++)
	{
		var[i] = 0;
	}
}

void scaleFeat(DescriptorF32 &accum, float scale)
{
	for(int i = 0; i < 128; i++)
	{
		if(scale > 0.01)
		{
			accum[i] /= scale;
		}
		else
		{
			accum[i] = 0;
		}
	}
}

double compareFeat(DescriptorF32 f1, DescriptorF32 f2)
{
	double diff = 0;

	for(int i = 0; i < 128; i++)
	{
		diff += fabs(f1[i] - f2[i]);
	}

	return diff;
}

void assignFeat(DescriptorF32 &f1, DescriptorF32 f2)
{
	for(int i = 0; i < 128; i++)
	{
		f1[i] = f2[i];
	}
}
