#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


__global__ void parallelTraslation(int* deviceChangeX, int* deviceChangeY, int* trn, int* trc, int* tra, int T)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= T)
		return;
	else {
		if (trc[idx] == 0) atomicSub(&deviceChangeX[trn[idx]], tra[idx]);
		else if (trc[idx] == 1) atomicAdd(&deviceChangeX[trn[idx]], tra[idx]);
		else if (trc[idx] == 2) atomicSub(&deviceChangeY[trn[idx]], tra[idx]);
		else if (trc[idx] == 3)atomicAdd(&deviceChangeY[trn[idx]], tra[idx]);
	}
}





__global__ void update_level(int* deviceChangeX, int* deviceChangeY, int* deviceGlobalCoordinatesX, int* deviceGlobalCoordinatesY, int* queue, int qStart, int qEnd, int* qptr, int* deviceCsr, int* deviceOffset)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid + qStart >= qEnd)
		return;
	else {
		deviceGlobalCoordinatesY[queue[tid + qStart]] += deviceChangeY[queue[tid + qStart]];
		deviceGlobalCoordinatesX[queue[tid + qStart]] += deviceChangeX[queue[tid + qStart]];
		int newStart = atomicAdd(qptr, deviceOffset[queue[tid + qStart] + 1] - deviceOffset[queue[tid + qStart]]);
		int i = deviceOffset[queue[tid + qStart]];
		while (i < deviceOffset[queue[tid + qStart] + 1])
		{
			deviceChangeX[deviceCsr[i]] += deviceChangeX[queue[tid + qStart]];
			deviceChangeY[deviceCsr[i]] += deviceChangeY[queue[tid + qStart]];
			queue[newStart + i - deviceOffset[queue[tid + qStart]]] = deviceCsr[i];
			i++;
		}
	}
}

__global__ void finalWrite(int node, int* gjpeg, int* gop, int* devicemesh, int meshop, int* gX, int* gY, int meshY, int frameX, int frameY)
{

	int fr = blockIdx.x + gX[node];
	int fc = threadIdx.x + gY[node];
	if (fr >= frameX || fc >= frameY || fr < 0 || fc < 0)
		return;
	else
	{
		if (meshop < gop[(fr)*frameY + fc])
			return;
		else
		{
			gop[(fr)*frameY + fc] = meshop;
			gjpeg[(fr)*frameY + fc] = devicemesh[blockIdx.x * meshY + threadIdx.x];
		}

	}
}

void readFile(const char* fileName, std::vector<SceneNode*>& scenes, std::vector<std::vector<int> >& edges, std::vector<std::vector<int> >& translations, int& frameSizeX, int& frameSizeY) {
	/* Function for parsing input file*/

	FILE* inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen(fileName, "r")) == NULL) {
		printf("Failed at opening the file %s\n", fileName);
		return;
	}

	// Input the header information.
	int numMeshes;
	fscanf(inputFile, "%d", &numMeshes);
	fscanf(inputFile, "%d %d", &frameSizeX, &frameSizeY);


	// Input all meshes and store them inside a vector.
	int meshX, meshY;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity;
	int* currMesh;
	for (int i = 0; i < numMeshes; i++) {
		fscanf(inputFile, "%d %d", &meshX, &meshY);
		fscanf(inputFile, "%d %d", &globalPositionX, &globalPositionY);
		fscanf(inputFile, "%d", &opacity);
		currMesh = (int*)malloc(sizeof(int) * meshX * meshY);
		for (int j = 0; j < meshX; j++) {
			for (int k = 0; k < meshY; k++) {
				fscanf(inputFile, "%d", &currMesh[j * meshY + k]);
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode(i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity);
		scenes.push_back(scene);
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf(inputFile, "%d", &relations);
	int u, v;
	for (int i = 0; i < relations; i++) {
		fscanf(inputFile, "%d %d", &u, &v);
		edges.push_back({ u,v });
	}

	// Input all translations.
	int numTranslations;
	fscanf(inputFile, "%d", &numTranslations);
	std::vector<int> command(3, 0);
	for (int i = 0; i < numTranslations; i++) {
		fscanf(inputFile, "%d %d %d", &command[0], &command[1], &command[2]);
		translations.push_back(command);
	}
}


void writeFile(const char* outputFileName, int* hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE* outputFile = NULL;
	if ((outputFile = fopen(outputFileName, "w")) == NULL) {
		printf("Failed while opening output file\n");
	}

	for (int i = 0; i < frameSizeX; i++) {
		for (int j = 0; j < frameSizeY; j++) {
			fprintf(outputFile, "%d ", hFinalPng[i * frameSizeY + j]);
		}
		fprintf(outputFile, "\n");
	}
}


int main(int argc, char** argv) {

	// Read the scenes into memory from File.
	const char* inputFileName = argv[1];
	int* hFinalPng;

	int frameSizeX, frameSizeY;
	std::vector<SceneNode*> scenes;
	std::vector<std::vector<int> > edges;
	std::vector<std::vector<int> > translations;
	readFile(inputFileName, scenes, edges, translations, frameSizeX, frameSizeY);
	hFinalPng = (int*)malloc(sizeof(int) * frameSizeX * frameSizeY);

	// Make the scene graph from the matrices.
	Renderer* scene = new Renderer(scenes, edges);

	// Basic information.
	int V = scenes.size();
	int E = edges.size();
	int numTranslations = translations.size();

	// Convert the scene graph into a csr.
	scene->make_csr(); // Returns the Compressed Sparse Row representation for the graph.
	int* hOffset = scene->get_h_offset();
	int* hCsr = scene->get_h_csr();
	int* hOpacity = scene->get_opacity(); // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int** hMesh = scene->get_mesh_csr(); // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int* hGlobalCoordinatesX = scene->getGlobalCoordinatesX(); // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int* hGlobalCoordinatesY = scene->getGlobalCoordinatesY(); // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int* hFrameSizeX = scene->getFrameSizeX(); // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int* hFrameSizeY = scene->getFrameSizeY(); // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now();


	// Code begins here.
	// Do not change anything above this comment.
	int* deviceCsr;
	cudaMalloc(&deviceCsr, sizeof(int) * E);
	cudaMemcpy(deviceCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);

	int* deviceOffset;
	cudaMalloc(&deviceOffset, sizeof(int) * (V + 1));
	cudaMemcpy(deviceOffset, hOffset, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);

	int* DeviceGlobalCoordinatesX;

	int htrn[numTranslations], htrc[numTranslations], htra[numTranslations];
	int i = 0;

	cudaMalloc(&DeviceGlobalCoordinatesX, sizeof(int) * (V));
	cudaMemcpy(DeviceGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	int* deviceChangeX;
	cudaMalloc(&deviceChangeX, sizeof(int) * V);
	cudaMemset(deviceChangeX, 0, sizeof(int) * V);

	int* deviceChangeY;
	cudaMalloc(&deviceChangeY, sizeof(int) * V);
	cudaMemset(deviceChangeY, 0, sizeof(int) * V);


	int* deviceGlobalCoordinatesY;
	cudaMalloc(&deviceGlobalCoordinatesY, sizeof(int) * (V));
	cudaMemcpy(deviceGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);

	while (i < numTranslations)
	{
		htrn[i] = translations[i][0];
		htrc[i] = translations[i][1];
		htra[i] = translations[i][2];
		i++;
	}

	int* trn, * trc, * tra;
	cudaMalloc(&trn, sizeof(int) * numTranslations);
	cudaMemcpy(trn, htrn, sizeof(int) * numTranslations, cudaMemcpyHostToDevice);

	cudaMalloc(&trc, sizeof(int) * numTranslations);
	cudaMemcpy(trc, htrc, sizeof(int) * numTranslations, cudaMemcpyHostToDevice);
	cudaMalloc(&tra, sizeof(int) * numTranslations);
	cudaMemcpy(tra, htra, sizeof(int) * numTranslations, cudaMemcpyHostToDevice);

	int numblocks_kernel1 = ceil(1.0 * numTranslations / 1024);
	int numthreadsperblock = 1024;
	if (numblocks_kernel1 == 0)
	{
		numblocks_kernel1 = 1;
		numthreadsperblock = numTranslations;
	}
	parallelTraslation << <numblocks_kernel1, numthreadsperblock >> > (deviceChangeX, deviceChangeY, trn, trc, tra, numTranslations);
	cudaError_t err1 = cudaGetLastError();
	cudaDeviceSynchronize();

	cudaFree(tra);
	cudaFree(trc);
	cudaFree(trn);

	int* gqueue;
	cudaMalloc(&gqueue, V * sizeof(int));
	cudaMemset(gqueue, 0, sizeof(int) * V);
	int* hqueuetop;
	hqueuetop = (int*)malloc(sizeof(int));
	*hqueuetop = 1;


	int* gqueuetop;
	cudaMalloc(&gqueuetop, sizeof(int));
	cudaMemcpy(gqueuetop, hqueuetop, sizeof(int), cudaMemcpyHostToDevice);

	int* hqueueback;
	hqueueback = (int*)malloc(sizeof(int));
	*hqueueback = 0;

	while ((*hqueuetop) - (*hqueueback) > 0)
	{
		update_level << < ((*hqueuetop - *hqueueback) + 1023) / 1024, 1024 >> > (deviceChangeX, deviceChangeY, DeviceGlobalCoordinatesX, deviceGlobalCoordinatesY, gqueue, *hqueueback, *hqueuetop, gqueuetop, deviceCsr, deviceOffset);
		cudaDeviceSynchronize();
		*hqueueback = *hqueuetop;
		cudaMemcpy(hqueuetop, gqueuetop, sizeof(int), cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();

	cudaFree(gqueue);
	cudaFree(deviceCsr);
	cudaFree(deviceChangeY);
	cudaFree(gqueuetop);
	cudaFree(deviceChangeX);
	cudaFree(deviceOffset);

	int* gop;
	cudaMalloc(&gop, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(gop, -1, sizeof(int) * frameSizeX * frameSizeY);

	int* devicemesh;
	cudaMalloc(&devicemesh, sizeof(int) * 10000);
	cudaMemset(devicemesh, 0, sizeof(int) * 10000);

	int* gjpeg;
	cudaMalloc(&gjpeg, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(gjpeg, 0, sizeof(int) * frameSizeX * frameSizeY);


	int q = 0;
	while (q < V)
	{
		cudaMemcpy(devicemesh, hMesh[q], sizeof(int) * hFrameSizeX[q] * hFrameSizeY[q], cudaMemcpyHostToDevice);
		finalWrite << <hFrameSizeX[q], hFrameSizeY[q] >> > (q, gjpeg, gop, devicemesh, hOpacity[q], DeviceGlobalCoordinatesX, deviceGlobalCoordinatesY, hFrameSizeY[q], frameSizeX, frameSizeY);
		q++;
	}
	cudaMemcpy(hFinalPng, gjpeg, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
	


	// Do not change anything below this comment.
	// Code ends here.

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::micro> timeTaken = end - start;

	printf("execution time : %f\n", timeTaken);
	// Write output matrix to file.
	const char* outputFileName = argv[2];
	writeFile(outputFileName, hFinalPng, frameSizeX, frameSizeY);

}
