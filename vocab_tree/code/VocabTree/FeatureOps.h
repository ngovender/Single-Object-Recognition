#ifndef __FEATUREOPS_H
#define __FEATUREOPS_H

#include "Defines.h"

double distanceFromCentre(DescriptorF32 centre, DescriptorUI8 feature);
void accumulate(DescriptorF32 &acc, DescriptorUI8 feat);
void zeroFeat(DescriptorF32 &var);
void scaleFeat(DescriptorF32 &accum, float scale);
double compareFeat(DescriptorF32 f1, DescriptorF32 f2);
void assignFeat(DescriptorF32 &f1, DescriptorF32 f2);

#endif