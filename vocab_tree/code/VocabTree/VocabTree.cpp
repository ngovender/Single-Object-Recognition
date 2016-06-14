#include <iostream>
#include <vector>
#include <list>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Defines.h"
#include "FeatureOps.h"
#include "FileIO.h"
#include "KMeans.h"

using namespace std;

#define NUM_FEATURES_FOR_TREE		400000

ImageList imageList;
TreeNode *root;

int main(int argc, char* argv[])
{
	cout << "Loading features..." << endl;

	// Load and count features from images
	char file[150];
	unsigned int fCount = 0;
	for(int i = 2565; i <= 3400; i += 1)
	{
		sprintf(file, "..\\..\\..\\..\\Data\\SFM3\\Feature Files\\UNDIST_im%05u.dat", i);
		fCount += loadFeatureList(file, &imageList);
	}

	// Calculate the frame interval to limit the number of features
	int interval = fCount / NUM_FEATURES_FOR_TREE;
	if(interval == 0)
	{
		interval = 1;
	}

	// Assign features to root node
	root = new TreeNode;
	for(int j = 0; j < imageList.size(); j += interval)
	{
		cout << ".";
		for(int i = 0; i < imageList[j]->N; i++)
		{
			root->descList.push_back(&(imageList[j]->d[i]));
		}
	}
	cout << endl;
	
	//vector<ImageData *>::iterator imListIter = imageList.begin();
	//for(; imListIter != imageList.end(); imListIter++)
	//{
	//	cout << ".";
	//	for(int i = 0; i < (*imListIter)->N; i++)
	//	{
	//		root->descList.push_back(&((*imListIter)->d[i]));
	//	}
	//}
	//cout << endl;

	clusterNode(root, 0, 4);

	saveTree("8x5.tree", root);

	// Load the stored tree from the disk
	root = loadTree("8x5.tree");

	// Create the inverted files for each of the nodes
	populateInvertedFiles(root, &imageList);

	// Calculate the node entropy
	calcEntropy(root, imageList.size());

	// Normalise the document vectors
	normaliseVectors(root, imageList.size());

	saveTree("8x5-Entropy.tree", root);

//	cout << "Choose image number for similarity calculation: " << endl;

	// Calculate the similarity vector for image 1
	cout << "Calculating similarity..." << endl;

	FILE *fd = fopen("simresult.txt", "w");

	for(int i = 0; i < imageList.size(); i++)
	{
		cout << "Processing image " << i << " of " << imageList.size() << endl;
		double *sim = calcSimilarity(root, i, imageList.size());
		for(int j = 0; j < imageList.size(); j++)
		{
			fprintf(fd, "%1.3f ", sim[j]);
		}
		fprintf(fd, "\n");
	}

	fclose(fd);

//	double *sim = calcSimilarity(root, 20, imageList.size());

	// Output similarity to screen
//	for(int i = 0; i < imageList.size(); i++)
//	{
//		cout << imageList[i]->filename << " : " << sim[i] << endl;
//	}

	system("pause");

	return 0;
}

