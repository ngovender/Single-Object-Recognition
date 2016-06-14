#ifndef __KMEANS_H
#define __KMEANS_H

#include "Defines.h"

void chooseInitialCentres(TreeNode *cluster);
void clusterNode(TreeNode *cluster, int currLevel, int maxLevel);
int treeDepth(TreeNode *root);
void updateDocCount(TreeNode *root, ImageList *images);
void populateInvertedFiles(TreeNode *root, ImageList *images);
void calcEntropy(TreeNode *root, int documentCount);
void normaliseVectors(TreeNode *root, int documentCount);
double *calcSimilarity(TreeNode *root, int docId, int documentCount);

#endif