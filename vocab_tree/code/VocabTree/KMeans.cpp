#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include "KMeans.h"
#include "FeatureOps.h"

using namespace std;

void chooseInitialCentres(TreeNode *cluster)
{
	// If more clusters than data points, assign the first couple of 
	//  centres to the available data points and then set the rest to zero
	if(cluster->descList.size() < K)
	{
		for(int i = 0; i < cluster->descList.size(); i++)
		{
			if(i < cluster->descList.size())
			{
				for(int j = 0; j < 128; j++)
				{
					cluster->centres[i][j] = (float)(*cluster->descList[i])[j];
				}
			}
			else
			{
				zeroFeat(cluster->centres[i]);
			}
		}

		return;
	}

	// Create a random set of unique lookup indices
	srand(time(NULL));
	int N = cluster->descList.size();
	int idx[K];
	for(int i = 0; i < K; i++)
	{
		idx[i] = (int)(((double)rand() / (double)RAND_MAX * (N - 1)) + 0.5);
		for(int j = 0; j < i; j++)
		{
			if(idx[i] == idx[j])
			{
				i--;
				break;
			}
		}
	}

	// Assign the random cluster centres
	for(int i = 0; i < K; i++)
	{
		for(int j = 0; j < 128; j++)
		{
			cluster->centres[i][j] = (float)(*cluster->descList[idx[i]])[j];
		}
	}
}

void clusterNode(TreeNode *cluster, int currLevel, int maxLevel)
{
	DescriptorF32 accum[K];
	int featCount[K];
	double minDist;
	int minIdx;
	double localDist;
	double clusterMovement = 1e10;

	cout << "======================== Clustering a node at level: " << currLevel << endl;
	cout << "======================== Number of features in cluster: " << cluster->descList.size() << endl;
	
	// Select initial centres at random from the dataset without replacement
	chooseInitialCentres(cluster);

	// Iterate until clusters stabilise
	while(clusterMovement > 0.1)
	{
		// Clear the accumulator and counter registers
		for(int i = 0; i < K; i++)
		{
			zeroFeat(accum[i]);
			featCount[i] = 0;
		}

		// Assign each descriptor to a cluster centre (For each descriptor)
		for(int fid = 0; fid < cluster->descList.size(); fid++)
		{
			minDist = 1e10;
			minIdx = -1;

			// Iterate over the potential cluster centres to determine nearest
			for(int cid = 0; cid < K; cid++)
			{
				localDist = distanceFromCentre(cluster->centres[cid], *(cluster->descList[fid]));
				if(localDist < minDist)
				{
					minDist = localDist;
					minIdx = cid;
				}
			}

			// Add feature vector to the accumulator
			accumulate(accum[minIdx], *(cluster->descList[fid]));
			featCount[minIdx]++;
		}

		// Scale the descriptors and check for cluster movement
		clusterMovement = 0;
		for(int i = 0; i < K; i++)
		{
			scaleFeat(accum[i], (float)featCount[i]);
			clusterMovement += compareFeat(cluster->centres[i], accum[i]);
			assignFeat(cluster->centres[i], accum[i]);
//			cout << i << ":" << featCount[i] << endl;
		}

		cout << clusterMovement << endl;
//		cout << "-------------------------------------------" << endl;
	}

	if(currLevel < maxLevel)
	{
		cout << "Populating children..." << endl;

		// Populate the children - Allocate memory
		for(int i = 0; i < K; i++)
		{
			cluster->children[i] = new TreeNode;
			cluster->children[i]->parent = cluster;
			cluster->children[i]->weight = 0;
			cluster->children[i]->docCount = 0;
			cluster->children[i]->lastDocId = -1;

			for(int j = 0; j < K; j++)
			{
				cluster->children[i]->children[j] = 0;
			}
		}

		// Transfer data pointers to the children
		for(int fid = 0; fid < cluster->descList.size(); fid++)
		{
			minDist = 1e10;
			minIdx = -1;

			// Iterate over the potential cluster centres to determine nearest
			for(int cid = 0; cid < K; cid++)
			{
				localDist = distanceFromCentre(cluster->centres[cid], *(cluster->descList[fid]));
				if(localDist < minDist)
				{
					minDist = localDist;
					minIdx = cid;
				}
			}

			// Assign feature vector to the closest clusters child
			cluster->children[minIdx]->descList.push_back(cluster->descList[fid]);
		}

		// Cluster the children
		for(int i = 0; i < K; i++)
		{
			clusterNode(cluster->children[i], currLevel + 1, maxLevel);
		}
	}
}

// Return the depth of the current tree
int treeDepth(TreeNode *root)
{
	int depth = 1; // Root node provides the first level

	while(root->children[0])
	{
		root = root->children[0];
		depth++;
	}

	return depth;
}

int getClosestCluster(TreeNode *node, DescriptorUI8 desc)
{
	double minDist = 1e10;
	int minIdx = -1;
	double localDist;

	// Iterate over the cluster centres to determine nearest
	for(int cid = 0; cid < K; cid++)
	{
		localDist = distanceFromCentre(node->centres[cid], desc);
		if(localDist < minDist)
		{
			minDist = localDist;
			minIdx = cid;
		}
	}

	return minIdx;
}

void parseFeature(TreeNode *node, DescriptorUI8 desc, int imageId)
{
	// Update the document counters
	if(node->lastDocId != imageId)
	{
		node->lastDocId = imageId;
		node->docCount++;
	}

	// If children exist, pass descriptor on to nearest cluster
	if(node->children[0])
	{
		int id = getClosestCluster(node, desc);
		parseFeature(node->children[id], desc, imageId);
	}
}

// Parse the image descriptors through the tree updating the document counts
void updateDocCount(TreeNode *root, ImageList *images)
{
	// Iterate through the images in the image list
	for(int iid = 0; iid < images->size(); iid++)
	{
		cout << "Parsing image " << iid << " of " << images->size() << endl;

		// Iterate through the features stored in the image
		for(int fid = 0; fid < (*images)[iid]->N; fid++)
		{
			parseFeature(root, (*images)[iid]->d[fid], iid);
		}
		
	}

}

void parseFeatureIF(TreeNode *node, DescriptorUI8 desc, int imageId)
{
	// Check if this document has visited this node before
	if(node->invFile.size())
	{
		// If the last list element corresponds to this imageId, update the nodeCount;
		if(node->invFile[node->invFile.size() - 1].imageId == imageId)
		{
			node->invFile[node->invFile.size() - 1].nodeCount++;
		}
		else	// Otherwise add a new list element
		{
			ListElement el;
			el.imageId = imageId;
			el.nodeCount = 1;
			node->invFile.push_back(el);
		}
	}
	else	// If no elements exist, add a new list element
	{
		ListElement el;
		el.imageId = imageId;
		el.nodeCount = 1;
		node->invFile.push_back(el);
	}
	
	// If children exist, pass descriptor on to nearest cluster
	if(node->children[0] != NULL)
	{
		int id = getClosestCluster(node, desc);
		parseFeatureIF(node->children[id], desc, imageId);
	}
}

// Fill the inverted files in the tree
void populateInvertedFiles(TreeNode *root, ImageList *images)
{
	// Iterate through the images in the image list
	for(int iid = 0; iid < images->size(); iid++)
	{
		cout << "Parsing image " << iid << " of " << images->size() << " : " << (*images)[iid]->N << endl;

		// Iterate through the features stored in the image
		for(int fid = 0; fid < (*images)[iid]->N; fid++)
		{
			parseFeatureIF(root, (*images)[iid]->d[fid], iid);
		}
	}
}

// Calculate the entropy of each of the nodes in the tree (recursive)
void calcEntropy(TreeNode *root, int documentCount)
{
	// Calculate the node entropy
	if(root->invFile.size() > 0)
	{
		root->weight = log((double)documentCount / (double)root->invFile.size());
	}
	else
	{
		root->weight = 0;
	}

	// If the children exist, calculate their entropy too
	if(root->children[0])
	{
		for(int i = 0; i < K; i++)
		{
			calcEntropy(root->children[i], documentCount);
		}
	}
}

void updateLength(TreeNode *node, double *lengths)
{
	// Update the lengths register for all images referred to by
	//  this nodes inverted file
	for(int i = 0; i < node->invFile.size(); i++)
	{
		lengths[node->invFile[i].imageId] += (node->weight * node->invFile[i].nodeCount) * (node->weight * node->invFile[i].nodeCount);
	}

	// Recursively iterate through all nodes
	if(node->children[0])
	{
		for(int i = 0; i < K; i++)
		{
			updateLength(node->children[i], lengths);
		}
	}
}

void normaliseWeights(TreeNode *node, double *lengths)
{
	// Normalise all entries in the inverse file using the weights provided in lengths
	for(int i = 0; i < node->invFile.size(); i++)
	{
		node->invFile[i].nodeCount /= lengths[node->invFile[i].imageId];
	}

	// Recursively iterate through all the nodes
	if(node->children[0])
	{
		for(int i = 0; i < K; i++)
		{
			normaliseWeights(node->children[i], lengths);
		}
	}
}

void normaliseVectors(TreeNode *root, int documentCount)
{
	double *length = new double[documentCount];
	
	// Initialise the length array
	for(int i = 0; i < documentCount; i++)
	{
		length[i] = 0;
	}

	// Calculate the lengths of all the vectors
	updateLength(root, length);

	// Square-root all the lengths to provide the normalising factor
	for(int i = 0; i < documentCount; i++)
	{
		length[i] = sqrt(length[i]);
	}

	// Normalise the inverted file weights
	normaliseWeights(root, length);

	// Recalculate lengths of all vectors and display for debug purposes
	for(int i = 0; i < documentCount; i++)
	{
		length[i] = 0;
	}
	updateLength(root, length);
	for(int i = 0; i < documentCount; i++)
	{
		cout << i << " : " << length[i] << endl;
	}
}

void calcSim4Node(TreeNode *node, int docId, double *sim)
{
	const double p = 2;

	// Check if q_i is non-zero in this node
	double q_i = 0;
	for(int i = 0; i < node->invFile.size(); i++)
	{
		if(node->invFile[i].imageId == docId)
		{
			q_i = node->invFile[i].nodeCount * node->weight;
//			cout << "Found q_i at position " << i << " : " << q_i << endl;
			break;
		}
	}

	// If a reasonable contribution is expected
	if(q_i > 1e-10)
	{
		for(int i = 0; i < node->invFile.size(); i++)
		{
			int idx = node->invFile[i].imageId;
			double d_i = node->invFile[i].nodeCount * node->weight;

			//sim[idx] += (pow(fabs(q_i - d_i), p) - pow(fabs(q_i), p) - pow(fabs(d_i), p));
			sim[idx] -= 2 * (q_i * d_i);
		}
	}

	// Recursivly iterate through children
	if(node->children[0])
	{
		for(int i = 0; i < K; i++)
		{
			calcSim4Node(node->children[i], docId, sim);
		}
	}
}

double *calcSimilarity(TreeNode *root, int docId, int documentCount)
{
	// Initialise the similarity vector
	double *sim = new double[documentCount];
	for(int i = 0; i < documentCount; i++)
	{
		sim[i] = 2;
	}

	// Iterate through the tree and update the similarity vector
	calcSim4Node(root, docId, sim);

	return sim;
}