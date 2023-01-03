#include <memory>
#include <iostream>
#include <chrono>
#include <future>
#include <cuda/std/cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "noita_random.h"
#include "stb_hbwang.h"
#include "coalhax.h"
#include "jps.hh"

#define BLOCKSIZE 32

typedef unsigned char byte;

__device__ __constant__ unsigned long COLOR_PURPLE = 0x7f007f;
__device__ __constant__ unsigned long COLOR_BLACK = 0x000000;
__device__ __constant__ unsigned long COLOR_WHITE = 0xffffff;
__device__ __constant__ unsigned long COLOR_YELLOW = 0xffff00;
__device__ __constant__ unsigned long COLOR_COFFEE = 0xc0ffee;

__device__ __constant__ uint map_w = 0;
__device__ __constant__ uint map_h = 0;
__device__ __constant__ uint worldSeedStart = 0;
__device__ __constant__ uint worldSeedCount = 0;
__device__ __constant__ int worldX = 0;
__device__ __constant__ int worldY = 0;
__device__ __constant__ bool isCoalMines = 0;

void __syncthreads();
int atomicAdd(int* address, int val);

// pudy248 note: If more generation differences occur, this would be the place to start debugging.
__device__ __constant__ unsigned long blockedColors[16] = {
	0x00ac33, //load_pixel_scene3
	0x00ac64, //load_pixel_scene4
	0x00ac6e, //load_pixel_scene4_alt
	0x4e175e, //load_oiltank_alt
	0x692e94, //load_pixel_scene_wide
	0x70d79e, //load_gunpowderpool_01
	0x70d7a0, //load_gunpowderpool_03
	0x70d7a1, //load_gunpowderpool_04
	0x7868ff, //load_gunpowderpool_02
	0x822e5b, //load_pixel_scene_tall
	0x97ab00, //load_pixel_scene5
	0xc35700, //load_oiltank
	0xc800ff, //load_pixel_scene_alt
	0xc9d959, //load_pixel_scene5b
	0xff0080, //load_pixel_scene2
	0xff0aff, //load_pixel_scene
}; 

__device__
unsigned long createRGB(const byte r, const byte g, const byte b)
{
	return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
}

__device__
unsigned long getPos(const uint w, byte f, const uint x, const uint y)
{
	return w * y * f + f * x;
}

__device__
unsigned long getPixelColor(const byte* map, const uint w, const uint x, const uint y)
{
	unsigned long long pos = getPos(w, 3, x, y);
	byte r = map[pos];
	byte g = map[pos + 1];
	byte b = map[pos + 2];
	return createRGB(r, g, b);
}

__device__
void setPixelColor(byte* map, uint w, uint x, uint y, unsigned long color)
{
	unsigned long long pos = getPos(w, 3, x, y);
	byte r = ((color >> 16) & 0xff);
	byte g = ((color >> 8) & 0xff);
	byte b = ((color)&0xff);
	map[pos] = r;
	map[pos + 1] = g;
	map[pos + 2] = b;
}

__device__
void fill(byte* map,
			int w,
			int x1,
			int x2,
			int y1,
			int y2,
			long color)
{
	for (int x = x1; x <= x2; x++)
	{
		for (int y = y1; y <= y2; y++)
		{
			setPixelColor(map, w, x, y, color);
		}
	}
}

struct intPair {
	int x;
	int y;
};

__device__
void floodFill(byte* map,
	uint width,
	uint height,
	int initialX,
	int initialY,
	unsigned long fromColor,
	unsigned long toColor,
	byte* visited,
	intPair* stack)
{
	int stackPtr = 0;
	if (initialX < 0 || initialX >= width || initialY < 0 || initialY >= height)
	{
		return;
	}

	stack[stackPtr++] = {initialX, initialY};
	visited[getPos(width, 1, initialX, initialY)] = true;

	int filled = 0;

	while (stackPtr != 0)
	{
		auto pos = stack[--stackPtr];
		const int x = pos.x;
		const int y = pos.y;

		setPixelColor(map, width, x, y, toColor);
		filled++;

		auto tryNext = [&map, &width, &height, &visited, &fromColor, &toColor, &stackPtr, &stack](int nx, int ny)
		{
			if (nx < 0 || nx >= width || ny < 0 || ny >= height)
			{
			return;
			}

			unsigned long long p = getPos(width, 1, nx, ny);
			if (visited[p] == 1)
			{
			return;
			}

			unsigned long nc = getPixelColor(map, width, nx, ny);
			if (nc != fromColor || nc == toColor)
			{
			return;
			}

			visited[p] = 1;
			stack[stackPtr++] = { nx, ny };
		};
		tryNext(x - 1, y);
		tryNext(x + 1, y);
		tryNext(x, y - 1);
		tryNext(x, y + 1);
	}
}

__device__
void fillC0ffee(
	byte* map,
	uint world_seed,
	byte* visited,
	intPair* stack)
{
	NollaPrng rng = NollaPrng(0);
	rng.SetRandomFromWorldSeed(world_seed);
	for (int y = 0; y < map_h; y++)
	{
		for (int x = 0; x < map_w; x++)
		{
			long c = getPixelColor(map, map_w, x, y);
			if (c != COLOR_COFFEE)
			{
			continue;
			}
			long to = COLOR_BLACK;
			double f = rng.Next();
			if (f < 0.5) // BIOME_RANDOM_BLOCK_CHANCE
			{
			to = COLOR_WHITE;
			}
			floodFill(map, map_w, map_h, x, y, COLOR_COFFEE, to, visited, stack);
		}
	}
}

NollaPrng GetRNG(int map_w, uint world_seed)
{
	NollaPrng rng = NollaPrng();
	rng.SetRandomFromWorldSeed(world_seed);
	rng.Next();
	int length = (int)((unsigned long long)((long long)map_w * -0x2e8ba2e9) >> 0x20);
	int iters = ((length >> 1) - (length >> 0x1f)) * 0xb + ((uint)world_seed / 0xc) * -0xc +
				world_seed + map_w;
	if (0 < iters)
	{
		do
		{
			rng.Next();
			iters -= 1;
		} while (iters != 0);
	}
	return rng;
}

__device__
void doCoalMineHax(
	byte* map,
	int width,
	int height)
{
	for (int i = 0; i < width * height * 3; i += 3)
	{
		long pix = createRGB(coalmine_overlay[i], coalmine_overlay[i + 1], coalmine_overlay[i + 2]);
		if (pix == 0x4000)
		{ // green. Looks like air?
			//pudy248 note: is not actually air, this is the main rock portion of the overlay
			map[i] = 0xFF;
			map[i + 1] = 0xFF;
			map[i + 2] = 0xFF;
		}
		if (pix == 0x0040)
		{ // blue. Looks like air?
			map[i] = 0x00;
			map[i + 1] = 0x00;
			map[i + 2] = 0x00;
		}
		if (pix == 0xFEFEFE)
		{ // white. Stairs. rock_static_intro
			// But in the debug it's not shown?
			// map[i] = 0x0a;
			// map[i + 1] = 0x33;
			// map[i + 2] = 0x44;
			map[i] = 0xFF;
			map[i + 1] = 0xFF;
			map[i + 2] = 0xFF;
		}
	}
}


__device__
bool contains(const unsigned long arr[16], unsigned long val)
{ 
	for (int i = 0; i < 16; i++)
		if (arr[i] == val) return true;
	return false;
};

__device__
void blockOutRooms(
	byte* map,
	int width,
	int height)
{
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 0; x < width; x++)
		{
			long color = getPixelColor(map, width, x, y);
			if (!contains(blockedColors, color))
			{
			continue;
			}
			int startX = x + 1;
			int endX = x + 1;
			int startY = y + 1;
			int endY = y + 1;
			bool foundEnd = false;
			while (!foundEnd && endX < width)
			{
			long c = getPixelColor(map, width, endX, startY);
			if (c == COLOR_BLACK)
			{
				endX += 1;
				continue;
			};
			endX -= 1;
			foundEnd = true;
			}
			if (endX >= width)
			{
			endX = width - 1;
			}
			foundEnd = false;
			while (!foundEnd && endY < height)
			{
			long c = getPixelColor(map, width, startX, endY);
			if (c == COLOR_BLACK)
			{
				endY += 1;
				continue;
			};
			endY -= 1;
			foundEnd = true;
			}
			if (endY >= height)
			{
			endY = height - 1;
			}
			fill(map, width, startX, endX, startY, endY, COLOR_WHITE);
		}
	}
}

__device__
const int BIOME_PATH_FIND_WORLD_POS_MIN_X = 159;
__device__
const int BIOME_PATH_FIND_WORLD_POS_MAX_X = 223;
__device__
const int WORLD_OFFSET_Y = 14;
__device__
const int WORLD_OFFSET_X = 35;

__device__ __shared__ intPair stackCache[BLOCKSIZE * 4];
__device__ __shared__ byte stackSize[BLOCKSIZE];

class Search
{
public:
	byte* map;

	byte* visited;
	int* queueFree;
	intPair* queueMem;

	byte pathFound;
	int queueSize;
	int targetX;
	int targetY;

	__device__
	void findPath(int x, int y, int threadIdx)
	{
		//printf("start\n");
		while (queueSize > 0 && pathFound != 1)
		{
			intPair n = pullCache(threadIdx);
			//printf("inner1 #%i %i, %i\n", threadIdx, n.x, n.y);
			if (n.x != -1) {
				//printf("inner2 #%i\n", threadIdx);
				if (atTarget(n))
				{
					//printf("AtTarget(%i, %i)\n", n.x, n.y);
					pathFound = 1;
				}
				tryNext(n.x, n.y - 1, threadIdx);
				tryNext(n.x - 1, n.y, threadIdx);
				tryNext(n.x + 1, n.y, threadIdx);
				tryNext(n.x, n.y + 1, threadIdx);
				//printf("inner3 #%i\n", threadIdx);
			}
			pushCache(threadIdx);
		}
	}

	__device__ intPair pullCache(int threadIdx) {
		__syncthreads();
		int idx = -1;
		if (queueSize > threadIdx) {
			//printf("pull#%i\n", threadIdx);
			int numOccupied = 0;
			int idxCounter = 0;
			for (int i = 0; i <= (queueSize >> 5) + 4 && idx == -1; i++) {
				uint read = queueFree[i];
				//printf("read: %i\n", read);
				
				for (int j = 0; j < 32; j++) {
					//printf("bit #%i: %i\n", j, ((read >> j) & 1));
					if (((read >> j) & 1) == 1) {
					numOccupied++;
					if ((numOccupied - 1) % BLOCKSIZE == threadIdx) {
						idx = 32 * i + j;
					}
				}
				}
			}
		}
		//printf("idx %i\n", idx);
		__syncthreads();
		intPair ret = { -1, -1 };
		if (queueSize > threadIdx) {
			ret = queueMem[idx];
			atomicAdd(queueFree + (idx >> 5), -(1 << (idx % 32)));
			atomicAdd(&queueSize, -1);
		}
		__syncthreads();
		return ret;
	}
	
	__device__ void pushCache(int threadIdx) {
		__syncthreads();
		int numFree = 0;
		int idxCounter = 0;
		int idx[4];
		idx[0] = -1;
		idx[1] = -1;
		idx[2] = -1;
		idx[3] = -1;
		//printf("push #%i, %i\n", threadIdx, stackSize[threadIdx]);
		for (int i = 0; i < ((map_w + map_h) >> 5) && idxCounter < stackSize[threadIdx]; i++) {
			int read = queueFree[i];
			
			for (int j = 0; j < 32; j++) {
				if (((read >> 0) & 1) == 0) {
					if (numFree % BLOCKSIZE == threadIdx) {
						idx[idxCounter] = 32 * i + j;
						idxCounter++;
					}
					numFree++;
				}
			}
		}

		//printf("idx#%i: %i %i %i %i\n", threadIdx, idx[0], idx[1], idx[2], idx[3]);

		__syncthreads();

		for (int i = 0; i < stackSize[threadIdx]; i++) {
			queueMem[idx[i]] = stackCache[threadIdx * 4 + i];
			atomicAdd(queueFree + (idx[i] >> 5), 1 << (idx[i] % 32)); //
			atomicAdd(&queueSize, 1);
			//printf("pqf: %i\n", idx[i]);
		}
		stackSize[threadIdx] = 0;

		__syncthreads();
	}

	__device__
	void tryNext(int x, int y, int threadIdx)
	{
		if (x >= 0 && y >= 0 && x < map_w && y < map_h) {
			if (traversable(x, y) && visited[y * map_w + x] == 0)
			{
				setPixelColor(map, map_w, x, y, COLOR_PURPLE);
				visited[y * map_w + x] = 1;
				stackCache[threadIdx * 4 + stackSize[threadIdx]] = { x, y };
				stackSize[threadIdx]++;
				//printf("#%i cache %i (%i, %i)", threadIdx, stackSize[threadIdx], x, y);
			}
		}
	}

	__device__ bool traversable(int x, int y)
	{
		long c = getPixelColor(map, map_w, x, y);

		return c == COLOR_BLACK || c == 0x2f554c;
	}
	__device__ bool atTarget(intPair n)
	{
		return targetY == n.y;
	}
	__device__ int Manhattan(int x, int y)
	{
		int dx = abs(x - targetX);
		int dy = abs(y - targetY);
		return (dx + dy);
	}
	__device__ int ManhattanDown(int x, int y)
	{
		int dy = abs(y - targetY);
		return (dy);
	}
};

__device__ __shared__ Search dSearch;

__device__
bool isMainPath()
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	return fill_x_to > 0 && fill_x_from > 0 && map_w > fill_x_from && fill_x_to < map_w + fill_x_from;
}

__device__
int fillMainPath(
	byte* map)
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	fill(map, map_w, fill_x_from, fill_x_to, 0, 6, COLOR_BLACK);
	return fill_x_from;
}


static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
	}
}


__global__
void blockRoomBlock(
	byte* block,
	byte* validBlock)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (!validBlock[idx]) {
			byte* segment = block + idx * (3 * map_w * map_h);
			blockOutRooms(segment, map_w, map_h);
		}
	}
}


__global__
void blockFillC0FFEE(
	byte* block,
	byte* validBlock,
	byte* visitedBlock,
	intPair* stackBlock)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (!validBlock[idx]) {
			byte* segment = block + idx * (3 * map_w * map_h);
			byte* visited = visitedBlock + idx * (map_w * map_h);
			intPair* stack = stackBlock + idx * 2 * (map_w + map_h);
			uint worldSeed = worldSeedStart + idx;
			fillC0ffee(segment, worldSeed, visited, stack);
		}
	}
}

__global__
void blockIsValid(
	byte* mapBlock,
	byte* validBlock,
	byte* sVisitedBlock,
	intPair* dQueueMem)
{	
	extern __shared__ int stackOccupied[];
	uint index = blockIdx.x;
	uint stride = gridDim.x;
	bool mainPath = isMainPath();

	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (!validBlock[idx]) {
			uint worldSeed = worldSeedStart + idx;
			byte* mapSegment = mapBlock + idx * 3 * map_w * map_h;
			byte* visitedSegment = sVisitedBlock + idx * map_w * map_h;
			intPair* queueMemSegment = dQueueMem + idx * 2 * (map_w + map_h);

			for (int i = 0; i < 10; i += 3) {
				mapSegment[i] = 0xff;
				mapSegment[i + 1] = 0;
				mapSegment[i + 2] = 0xff;
			}

			
			uint path_start_x = 0;
			if (mainPath)
			{
				if (isCoalMines)
				{
					path_start_x = 0x8e;
				}
				else
				{
					path_start_x = fillMainPath(mapSegment);
				}
			}

			bool hasPath;
			int x = path_start_x;

			if (!mainPath) {
				while (x < map_w)
				{
					long c = getPixelColor(mapSegment, map_w, x, 0);
					if (c != COLOR_BLACK)
					{
						x++;
						continue;
					}
				}
			}
			if (threadIdx.x == 0) {
				//printf("--#%i\n", idx);
				dSearch.map = mapSegment;
				dSearch.visited = visitedSegment;
				dSearch.queueMem = queueMemSegment;
				dSearch.queueFree = stackOccupied;
				dSearch.queueSize = 1;

				dSearch.targetX = x;
				dSearch.targetY = map_h - 1;

				dSearch.visited[x] = 1;
				setPixelColor(dSearch.map, map_w, x, 0, COLOR_PURPLE);
				stackOccupied[0] = 1;
				queueMemSegment[0] = { x, 0 };

			}
			stackSize[threadIdx.x] = 0;
			stackCache[threadIdx.x * 4] = { -1,-1 };
			stackCache[threadIdx.x * 4 + 1] = { -1,-1 };
			stackCache[threadIdx.x * 4 + 2] = { -1,-1 };
			stackCache[threadIdx.x * 4 + 3] = { -1,-1 };
			__syncthreads();
			
			dSearch.findPath(path_start_x, 0, threadIdx.x);
			
			if (threadIdx.x == 0) {
				validBlock[idx] = dSearch.pathFound;
			}
			
			__syncthreads();
		}
	}
}

__global__ 
void blockCoalMineHax(
	byte* block, 
	byte* validBlock, 
	char checkValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (!checkValid || !validBlock[idx]) {
			byte* segment = block + (idx * (3 * map_w * map_h));
			doCoalMineHax(segment, map_w, map_h);
		}
	}
}

STBHW_EXTERN{
	__declspec(dllexport) void free_array(byte* block) {
		free(block);
	}
}


STBHW_EXTERN{
	__declspec(dllexport) byte* generate_block(
	byte host_tileData[],
		uint tiles_w,
		uint tiles_h,
		uint _map_w,
		uint _map_h,
		bool _isCoalMine,
		int _worldX,
		int _worldY,
		uint _worldSeedStart,
		uint _worldSeedCount) {

		printf("DLL 1\n");

		checkCudaErrors(cudaMemcpyToSymbol(map_w, &_map_w, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(map_h, &_map_h, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(isCoalMines, &_isCoalMine, sizeof(bool)));
		checkCudaErrors(cudaMemcpyToSymbol(worldX, &_worldX, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(worldY, &_worldY, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(worldSeedStart, &_worldSeedStart, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(worldSeedCount, &_worldSeedCount, sizeof(uint)));

		int numBlocks = (int)((float)_worldSeedCount / BLOCKSIZE) + 1;
		byte* resultBlock = (byte*)malloc(3 * _map_w * _map_h * _worldSeedCount);
		byte* resBlock = (byte*)malloc(3 * _map_w * (_map_h + 4) * _worldSeedCount);
		byte* validBlock = (byte*)malloc(_worldSeedCount);
		NollaPrng* rngBlock1 = (NollaPrng*)malloc(sizeof(NollaPrng) * _worldSeedCount);
		NollaPrng* rngBlock2 = (NollaPrng*)malloc(sizeof(NollaPrng) * _worldSeedCount);

		byte* dResultBlock;
		byte* dValidBlock;
		byte* dVisitedBlock;
		intPair* dStackMem;

		checkCudaErrors(cudaMalloc((void**)&dResultBlock, 3 * _map_w * _map_h * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dValidBlock, _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dVisitedBlock, _worldSeedCount * _map_w * _map_h));
		checkCudaErrors(cudaMalloc((void**)&dStackMem, sizeof(intPair) * _worldSeedCount * 2 * (_map_w + _map_h)));

		checkCudaErrors(cudaMemset(dValidBlock, 0, _worldSeedCount));

		stbhw_tileset tileSet;
		stbhw_build_tileset_from_image(&tileSet, host_tileData, tiles_w * 3, tiles_w, tiles_h);

		for (int i = 0; i < _worldSeedCount; i++) {
			uint worldSeed = _worldSeedStart + i;

			rngBlock1[i] = GetRNG(_map_w, worldSeed);
			rngBlock2[i] = NollaPrng(0);
		}

		bool stop = false;

		long long int mapgenTime = 0;
		long long int miscTime = 0;
		long long int validateTime = 0;

		int tries = 0;
		while (!stop) {
			if (tries > 99) break;
			chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

			for (int i = 0; i < _worldSeedCount; i++) {
				if (tries == 0 || !validBlock[i]) {
					byte* res = resBlock + i * (3 * _map_w * (_map_h + 4));
					byte* result = resultBlock + i * (3 * _map_w * _map_h);

					rngBlock2[i].Seed = rngBlock1[i].Next() * 2147483645.0;
					rngBlock2[i].Next();

					function<int()> func = bind(&NollaPrng::NextU, &(rngBlock2[i]));

					stbhw_generate_image(&tileSet, NULL, res, _map_w * 3, _map_w, _map_h + 4, func);
				}
			}

			for (int i = 0; i < _worldSeedCount; i++) {
				if(tries == 0 || !validBlock[i])
					memcpy(resultBlock + i * 3 * _map_w * _map_h, resBlock + i * 3 * _map_w * (_map_h + 4) + 4 * 3 * _map_w, 3 * _map_w * _map_h);
			}

			chrono::steady_clock::time_point time2 = chrono::steady_clock::now();

			checkCudaErrors(cudaMemcpy(dResultBlock, resultBlock, 3 * _map_w * _map_h * _worldSeedCount, cudaMemcpyHostToDevice));

			checkCudaErrors(cudaMemset(dVisitedBlock, 0, _worldSeedCount * _map_w * _map_h));
			blockFillC0FFEE<<<numBlocks, BLOCKSIZE>>>(dResultBlock, dValidBlock, dVisitedBlock, dStackMem);
			checkCudaErrors(cudaDeviceSynchronize());

			if (_worldY < 20 && _worldX > 32 && _worldX < 39) {
				blockRoomBlock<<<numBlocks, BLOCKSIZE>>>(dResultBlock, dValidBlock);
				checkCudaErrors(cudaDeviceSynchronize());
			}

			if (_isCoalMine) {
				blockCoalMineHax<<<numBlocks, BLOCKSIZE>>>(dResultBlock, dValidBlock, true);
				checkCudaErrors(cudaDeviceSynchronize());
			}


			chrono::steady_clock::time_point time3 = chrono::steady_clock::now();
			
			checkCudaErrors(cudaMemset(dVisitedBlock, 0, _worldSeedCount * _map_w * _map_h));
			printf("DLL 2\n");
			blockIsValid<<<numBlocks, BLOCKSIZE, (_map_w + _map_h >> 5) + 8>>>(dResultBlock, dValidBlock, dVisitedBlock, dStackMem);
			checkCudaErrors(cudaDeviceSynchronize());
			printf("DLL 3\n");

			checkCudaErrors(cudaMemcpy(validBlock, dValidBlock, _worldSeedCount, cudaMemcpyDeviceToHost));

			chrono::steady_clock::time_point time4 = chrono::steady_clock::now();
			mapgenTime += chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();
			miscTime += chrono::duration_cast<chrono::milliseconds>(time3 - time2).count();
			validateTime += chrono::duration_cast<chrono::milliseconds>(time4 - time3).count();


			tries++;
			stop = true;
			//for (int j = 0; j < _worldSeedCount; j++) if (!validBlock[j]) { stop = false; break; }
		}

		//checkCudaErrors(cudaMemcpy(dResultBlock, resultBlock, 3 * _map_w * _map_h * _worldSeedCount, cudaMemcpyHostToDevice));
		
		//if (_isCoalMine) {
		//	blockCoalMineHax<<<numBlocks, BLOCKSIZE>>>(dResultBlock, dValidBlock, false);
		//	checkCudaErrors(cudaDeviceSynchronize());
		//}

		checkCudaErrors(cudaMemcpy(resultBlock, dResultBlock, 3 * _map_w * _map_h * _worldSeedCount, cudaMemcpyDeviceToHost));

		cudaFree(dResultBlock);
		cudaFree(dValidBlock);
		cudaFree(dVisitedBlock);
		cudaFree(dStackMem);

		free(resBlock);
		free(validBlock);
		free(rngBlock1);
		free(rngBlock2);

		printf("DLL 9\n");

		printf("WORLDGEN ACCUMULATED TIME: %lli ms\n", mapgenTime);
		printf("VALIDATE ACCUMULATED TIME: %lli ms\n", validateTime);
		printf("MISCELL. ACCUMULATED TIME: %lli ms\n", miscTime);

		return resultBlock;
	}
}

__device__
int GetWidthFromPix(int a, int b)
{
	return ((b * 512) / 10 - (a * 512) / 10);
}

__device__
int GetGlobalPos(int a, int b, int c)
{
	return ((b * 512) / 10 - (a * 512) / 10) * 10 + c;
}

int main() {
	return 0;
}