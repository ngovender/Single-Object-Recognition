#include <iostream>
#include <vector>
#include "FileIO.h"
#include "KMeans.h"

using namespace std;

//#define PATH		"D:\\PhD Work\\Data\\Loop3Times\\FeatureFiles\\Random\\"
#define PATH		""

int loadFeatureList(const char *filename, ImageList *imageList)
{
	char fullPath[150] = PATH;
	ImageData *image = new ImageData;
	FILE *fd = fopen(strcat(fullPath, filename), "rb");

	// Save the filename
	strncpy(image->filename, filename, 20);

	// Read the number of stored feature vectors
	image->N = 0;
	fread(&image->N, 2, 1, fd);
	cout << "Reading " << image->N << " features for image " << filename <<endl;

	// Read the feature vectors
	image->d = new DescriptorUI8[image->N];
	fread(image->d, 128, image->N, fd);

	// Close the file
	fclose(fd);

	// Add the image data to the image list
	imageList->push_back(image);

	return image->N;
}

// Recursively save the node data to the file
void saveNode(FILE *fd, TreeNode *node)
{
	cout << ".";
	fwrite(node->centres, sizeof(DescriptorF32), K, fd);
	fwrite(&node->weight, sizeof(node->weight), 1, fd);

	if(node->children[0])
	{
		for(int i = 0; i < K; i++)
		{
			saveNode(fd, node->children[i]);
		}
	}
}

void loadNode(FILE *fd, TreeNode *node)
{
	fread(node->centres, sizeof(DescriptorF32), K, fd);
	fread(&node->weight, sizeof(node->weight), 1, fd);

	if(node->children[0])
	{
		for(int i = 0; i < K; i++)
		{
			loadNode(fd, node->children[i]);
		}
	}
}

void saveTree(const char *filename, TreeNode *root)
{
	// Open file for saving the tree
	char fullPath[150] = PATH;
	FILE *fd = fopen(strcat(fullPath, filename), "wb");
	if(!fd)
	{
		cout << "Error writing to file: " << fullPath << endl;
		system("pause");
		exit(1);
	}

	// Store the tree parameters to the file
	unsigned short data = K;
	fwrite(&data, sizeof(data), 1, fd);
	data = treeDepth(root);
	fwrite(&data, sizeof(data), 1, fd);

	cout << "Saving tree to file with branch factor " << K;
	cout << " and depth " << data << endl;

	// Save the tree recursively to the disk
	saveNode(fd, root);
	cout << endl;

	// Close the file
	fclose(fd);
}

TreeNode *makeTree(TreeNode *parent, int depth)
{
	// Create the new node and assign the parent node
	TreeNode *node = new TreeNode;
	node->parent = parent;
	node->weight = 0;
	node->docCount = 0;
	node->lastDocId = -1;

	// Populate child nodes
	for(int i = 0; i < K; i++)
	{
		// If at leaf nodes, set children to 0, otherwise add sub-tree
		if(depth == 1)
		{
			node->children[i] = 0;
		}
		else
		{
			node->children[i] = makeTree(node, depth - 1);
		}
	}

	return node;
}

TreeNode *loadTree(const char *filename)
{
	// Open file for loading the tree
	char fullPath[150] = PATH;
	FILE *fd = fopen(strcat(fullPath, filename), "rb");
	if(!fd)
	{
		cout << "Error writing to file: " << fullPath << endl;
		system("pause");
		exit(1);
	}

	// Load the tree parameters
	unsigned short k, d;
	fread(&k, sizeof(k), 1, fd);
	fread(&d, sizeof(k), 1, fd);

	// Validate the branch factor
	if(k != K)
	{
		cout << "Unable to load a tree of size " << k << endl;
		system("pause");
		exit(1);
	}

	cout << "Loading a tree with branch factor " << k << " and depth " << d << endl;

	// Create the empty tree and populate it
	TreeNode *root = makeTree(NULL, d);
	loadNode(fd, root);
	cout << endl;

	// Return the newly created tree
	return root;
}