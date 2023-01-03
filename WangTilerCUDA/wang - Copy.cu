#include <memory>
#include <iostream>
#include <chrono>
#include <cuda/std/cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "noita_random.h"
#include "stb_hbwang.h"
#include "coalhax.h"

//TODO fiddle with these to maximize performance, not sure what the correct configuration is
#define NUMBLOCKS 512
#define BLOCKSIZE 64

typedef unsigned char byte;

__device__ __constant__ unsigned long COLOR_PURPLE = 0x7f007f;
__device__ __constant__ unsigned long COLOR_BLACK00 = 0x000000;
__device__ __constant__ unsigned long COLOR_BLACK01 = 0x010101;
__device__ __constant__ unsigned long COLOR_BLACK02 = 0x020202;
__device__ __constant__ unsigned long COLOR_WHITE = 0xffffff;
__device__ __constant__ unsigned long COLOR_YELLOW = 0xffff00;
__device__ __constant__ unsigned long COLOR_COFFEE = 0xc0ffee;

__device__ __constant__ uint map_w = 0;
__device__ __constant__ uint map_h = 0;
__device__ __constant__ int worldX = 0;
__device__ __constant__ int worldY = 0;
__device__ __constant__ bool isCoalMines = 0;
__device__ __constant__ uint worldSeedStart = 0;
__device__ __constant__ uint worldSeedCount = 0;
__device__ __constant__ int pwCount = 0;
__device__ __constant__ byte ngPlus = 0;
__device__ __constant__ uint maxChestContents;
__device__ __constant__ uint maxChestsPerWorld;
__device__ __constant__ byte loggingLevel = 0;

// pudy248 note: If more generation differences occur, this would be the place to start debugging.
#define BCSize 16
__device__ __constant__ unsigned long blockedColors[BCSize] = {
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

//GPU memory doesn't like integers which aren't aligned to 4-byte boundaries, so we have to use these methods for memory accesses.
//Technically, since the CPU code just casts to int*, hard-coding a little-endian format may cause errors, but we'll deal with that when we get there.
__device__ int readUnalignedInt(byte* ptr) {
	//some of these casts and parentheses may be unnecessary but better safe than sorry.
	return 
		(((int)*(signed char*)(ptr + 3)) << 24) | 
		(*(ptr + 2) << 16) | 
		(*(ptr + 1) << 8) | 
		*(ptr + 0);
}

__device__ void writeUnalignedInt(byte* ptr, int val) {
	*(ptr + 3) = val >> 24;
	*(ptr + 2) = (val >> 16) & 0xff;
	*(ptr + 1) = (val >> 8) & 0xff;
	*(ptr + 0) = val & 0xff;
}

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
	rng.SetRandomFromWorldSeed(world_seed + ngPlus);
	rng.Next();
	for (int y = 0; y < map_h; y++)
	{
		for (int x = 0; x < map_w; x++)
		{
			long c = getPixelColor(map, map_w, x, y);
			if (c != COLOR_COFFEE)
			{
				continue;
			}
			long to = COLOR_BLACK00;
			double f = rng.Next();
			if (f <= 0.5) // BIOME_RANDOM_BLOCK_CHANCE
			{
				to = COLOR_WHITE;
			}
			floodFill(map, map_w, map_h, x, y, COLOR_COFFEE, to, visited, stack);
		}
	}
}

__device__
NollaPrng GetRNG(int map_w, uint world_seed)
{
	uint wsAdjusted = world_seed + ngPlus;
	NollaPrng rng = NollaPrng();
	rng.SetRandomFromWorldSeed(wsAdjusted);
	rng.Next();
	int length = (int)((unsigned long long)((long long)map_w * -0x2e8ba2e9) >> 0x20);
	int iters = ((length >> 1) - (length >> 0x1f)) * 0xb + ((uint)wsAdjusted / 0xc) * -0xc +
		wsAdjusted + map_w;
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
	int phase,
	int width,
	int height)
{
	for (int i = 0; i < width * height * 3; i += 3)
	{
		long pix = createRGB(coalmine_overlay[i], coalmine_overlay[i + 1], coalmine_overlay[i + 2]);
		if (phase == 2 && pix == 0x4000)
		{ // green. Looks like air?
			//pudy248 note: is not actually air, this is the main rock portion of the overlay
			map[i] = 0xFF;
			map[i + 1] = 0xFF;
			map[i + 2] = 0xFF;
		}
		if (phase == 1 && pix == 0x0040)
		{ // blue. Looks like air?
			map[i] = 0x00;
			map[i + 1] = 0x00;
			map[i + 2] = 0x00;
		}
		if (phase == 1 && pix == 0xFEFEFE)
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
bool contains(const unsigned long arr[BCSize], unsigned long val)
{ 
	for (int i = 0; i < BCSize; i++)
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
			if (c == COLOR_BLACK00)
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
			if (c == COLOR_BLACK00)
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
const int WORLD_OFFSET_X = 35;

__device__ __shared__ intPair stackCache[BLOCKSIZE * 4];
__device__ __shared__ byte stackSize[BLOCKSIZE];

class Search
{
public:
	byte* map;

	byte* visited;
	intPair* queueMem;

	byte pathFound;
	int queueSize;
	int targetX;
	int targetY;
	int threadIdx;

	__device__
	void findPath(int x, int y)
	{
		while (queueSize > 0 && pathFound != 1)
		{
			stackSize[threadIdx] = 0;
			intPair n = Pop();
			setPixelColor(map, map_w, n.x, n.y, COLOR_PURPLE);
			if (n.x != -1) {
				if (atTarget(n))
				{
					pathFound = 1;
				}
				tryNext(n.x, n.y + 1);
				tryNext(n.x - 1, n.y);
				tryNext(n.x + 1, n.y);
				tryNext(n.x, n.y - 1);
			}
			Push();
		}
	}

	__device__ intPair Pop() {
		return queueMem[--queueSize];
	}
	
	__device__ void Push() {
		while (stackSize[threadIdx] > 0) {
			queueMem[queueSize++] = stackCache[4 * threadIdx + --stackSize[threadIdx]];
		}
	}

	__device__
	void tryNext(int x, int y)
	{
		if (x >= 0 && y >= 0 && x < map_w && y < map_h) {
			if (traversable(x, y) && visited[y * map_w + x] == 0)
			{
				visited[y * map_w + x] = 1;
				stackCache[threadIdx * 4 + stackSize[threadIdx]] = { x, y };
				stackSize[threadIdx]++;
			}
		}
	}

	__device__ bool traversable(int x, int y)
	{
		long c = getPixelColor(map, map_w, x, y);

		return c == COLOR_BLACK00 || c == COLOR_BLACK02 || c == 0x2f554c;
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

__device__ __shared__ Search dSearch[BLOCKSIZE];

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
	fill(map, map_w, fill_x_from, fill_x_to, 0, 6, COLOR_BLACK00);
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

__host__ __device__
int GetGlobalPosX(int x, int y, int px, int py)
{
	if (x == 14)
	{
		py -= 10;
	}
	int gx = (int)(((x - 35) * 512) / 10) * 10 + px - 15;
	//int gy = (int)(((y - 14) * 512) / 10) * 10 + py - 3;
	return gx;
}
__host__ __device__
int GetGlobalPosY(int x, int y, int px, int py)
{
	if (y == 14)
	{
		py -= 10;
	}
	//int gx = (int)(((x - 35) * 512) / 10) * 10 + px - 15;
	int gy = (int)(((y - 14) * 512) / 10) * 10 + py - 3;
	return gy;
}


// 0 gold_nuggets
// 1 chest_to_gold
// 2 rain_gold
// 3 bomb
// 4 powder
// 5 potion_normal
// 6 potion_secret
// 7 potion_random_material
// 8 potions_pps
// 9 potions_ssr
// 10 kammi
// 11 kuu
// 12 paha_silma
// 13 chaos_die
// 14 shiny_orb
// 15 ukkoskivi
// 16 kiuaskivi
// 17 vuoksikivi
// 18 kakkakikkare
// 19 runestone_light
// 20 runestone_fire
// 21 runestone_magma
// 22 runestone_weight
// 23 runestone_emptiness
// 24 runestone_edges
// 25 runestone_metal
// 26 random_spell
// 27 spell_refresh
// 28 heart_normal
// 29 heart_mimic
// 30 large_heart
// 31 full_heal
// 32 wand_T1
// 33 wand_T1NS
// 34 wand_T2
// 35 wand_T2NS
// 36 wand_T3
// 37 wand_T3NS
// 38 wand_T4
// 39 wand_T4NS
// 40 wand_T5
// 41 wand_T5NS
// 42 wand_T6
// 43 wand_T6NS
// 
// 254 sampo
// 255 orb

__device__ void CheckNormalChestLoot(int x, int y, uint worldSeed, byte* writeLoc)
{
	writeUnalignedInt(writeLoc, x);
	writeUnalignedInt(writeLoc + 4, y);
	byte* contents = writeLoc + 9;
	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x, y);

	int idx = 0;
	int count = 1;
	while (count > 0)
	{
		if (idx >= maxChestContents) {
			if(loggingLevel >= 3) printf("Chest contents overflow in seed %i!\n", worldSeed);
			break;
		}
		count--;
		int rnd = random.Random(1, 100);
		if (rnd <= 7) contents[idx++] = 3;
		else if (rnd <= 40) 
		{
			rnd = random.Random(0, 100);

			rnd = random.Random(0, 100);
			if (rnd > 99)
			{
				int tamount = random.Random(1, 3);
				for (int i = 0; i < tamount; i++)
				{
					random.Random(-10, 10);
					random.Random(-10, 5);
				}

				if (random.Random(0, 100) > 50)
				{
					tamount = random.Random(1, 3);
					for (int i = 0; i < tamount; i++) {
						random.Random(-10, 10);
						random.Random(-10, 5);
					}
				}
				if (random.Random(0, 100) > 80) {
					tamount = random.Random(1, 3);
					for (int i = 0; i < tamount; i++) {
						random.Random(-10, 10);
						random.Random(-10, 5);
					}
				}
			}
			else {
				random.Random(-10, 10);
				random.Random(-10, 5);
			}
			contents[idx++] = 0;
		}
		else if (rnd <= 50)
		{
			rnd = random.Random(1, 100);
			if (rnd <= 94) contents[idx++] = 5;
			else if (rnd <= 90) contents[idx++] = 4;
			else
			{
				random.Random(1, 100);
				if (rnd <= 98) contents[idx++] = 6;
				else contents[idx++] = 7;
			}
		}
		else if (rnd <= 54) contents[idx++] = 27;
		else if (rnd <= 60)
		{
			byte opts[8] = { 10, 11, 15, 12, 16, 127, 13, 14 };
			rnd = random.Random(0, 7);
			byte opt = opts[rnd];
			if (opt == 127)
			{

				byte r_opts [7] = {19, 20, 21, 22, 23, 24, 25};
				rnd = random.Random(0, 6);
				byte r_opt = r_opts[rnd];
				contents[idx++] = r_opt;
			}
			else
			{
				contents[idx++] = opt;
			}
		}
		else if (rnd <= 65) contents[idx++] = 26;
		else if (rnd <= 84)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) contents[idx++] =32;
			else if (rnd <= 50) contents[idx++] = 33;
			else if (rnd <= 75) contents[idx++] = 34;
			else if (rnd <= 90) contents[idx++] = 35;
			else if (rnd <= 96) contents[idx++] = 36;
			else if (rnd <= 98) contents[idx++] = 37;
			else if (rnd <= 99) contents[idx++] = 38;
			else contents[idx++] = 39;
		}
		else if (rnd <= 95)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 88) contents[idx++] = 28;
			else if (rnd <= 89) contents[idx++] = 29;
			else if (rnd <= 99) contents[idx++] = 30;
			else contents[idx++] = 31;
		}
		else if (rnd <= 98) contents[idx++] = 1;
		else if (rnd <= 99) count += 2;
		else count += 3;
	}
	*(writeLoc + 8) = (byte)idx;
}

__device__ void CheckGreatChestLoot(int x, int y, uint worldSeed, byte* writeLoc)
{
	writeUnalignedInt(writeLoc, x);
	writeUnalignedInt(writeLoc + 4, y);
	byte* contents = writeLoc + 9;
	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x, y);

	int idx = 0;
	int count = 1;

	if (random.Random(0, 100000) >= 100000)
	{
		count = 0;
		if (random.Random(0, 1000) == 999) contents[idx++] = 255;
		else contents[idx++] = 254;
	}

	while (count != 0)
	{
		if (idx >= maxChestContents) {
			if (loggingLevel >= 3) printf("Chest contents overflow in seed %i!\n", worldSeed);
			break;
		}
		count--;
		int rnd = random.Random(1, 100);

		if (rnd <= 30)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 30)
				contents[idx++] = 8;
			else
				contents[idx++] = 9;
		}
		else if (rnd <= 33)
		{
			contents[idx++] = 2;
		}
		else if (rnd <= 38)
		{
			rnd = random.Random(1, 30);
			if (rnd == 30)
			{
				contents[idx++] = 18;
			}
			else contents[idx++] = 17;
		}
		else if (rnd <= 39)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) contents[idx++] = 36;
			else if (rnd <= 50) contents[idx++] = 37;
			else if (rnd <= 75) contents[idx++] = 38;
			else if (rnd <= 90) contents[idx++] = 39;
			else if (rnd <= 96) contents[idx++] = 40;
			else if (rnd <= 98) contents[idx++] = 41;
			else if (rnd <= 99) contents[idx++] = 42;
			else contents[idx++] = 43;
		}
		else if (rnd <= 60)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 89) contents[idx++] = 28;
			else if (rnd <= 99) contents[idx++] = 30;
			else contents[idx++] = 31;
		}
		else if (rnd <= 99) count += 2;
		else count += 3;
	}
	*(writeLoc + 8) = (byte)idx;
}

__device__ void spawnHeart(int x, int y, uint seed, byte* writeLoc) {
	NoitaRandom random = NoitaRandom(seed + ngPlus);
	if (loggingLevel >= 5) printf("Spawning heart: %i, %i\n", x, y);
	float r = random.ProceduralRandomf(x, y, 0, 1);
	float heart_spawn_percent = 0.7f;

	if (r <= heart_spawn_percent && r > 0.3)
	{
		random.SetRandomSeed(x + 45, y - 2123);
		int rnd = random.Random(1, 100);
		if (rnd <= 90 || y < 512 * 3)
		{
			rnd = random.Random(1, 1000);

			if (rnd >= 1000)
				CheckGreatChestLoot(x, y, seed, writeLoc);
			else 
				CheckNormalChestLoot(x, y, seed, writeLoc);
		}
	}
	else {
		writeUnalignedInt(writeLoc, -1);
		writeUnalignedInt(writeLoc + 4, -1);
	}
}

__device__ void spawnChest(int x, int y, uint seed, byte greedCurse, byte* writeLoc)
{
	NoitaRandom random = NoitaRandom(seed + ngPlus);
	if(loggingLevel >= 5) printf("Spawning guaranteed chest: %i, %i\n", x, y);
	random.SetRandomSeed(x, y);
	int super_chest_spawn_rate = greedCurse > 0 ? 100 : 2000;
	int rnd = random.Random(1, super_chest_spawn_rate);

	if (rnd >= super_chest_spawn_rate - 1)
		CheckGreatChestLoot(x, y, seed, writeLoc);
	else
		CheckNormalChestLoot(x, y, seed, writeLoc);
}

__global__
void blockRoomBlock(
	byte* block,
	byte* validBlock,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
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
	intPair* stackBlock,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* segment = block + idx * (3 * map_w * map_h);
			byte* visited = visitedBlock + idx * (map_w * map_h);
			intPair* stack = stackBlock + idx * (map_w + map_h);
			uint worldSeed = worldSeedStart + idx;

			memset(visited, 0, map_w * map_h);

			fillC0ffee(segment, worldSeed, visited, stack);
		}
	}
}

__global__
void blockIsValid(
	byte* mapBlock,
	byte* validBlock,
	byte* sVisitedBlock,
	intPair* dQueueMem,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	bool mainPath = isMainPath();

	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* mapSegment = mapBlock + idx * 3 * map_w * map_h;
			byte* visitedSegment = sVisitedBlock + idx * map_w * map_h;
			intPair* queueMemSegment = dQueueMem + idx * (map_w + map_h);

			memset(visitedSegment, 0, worldSeedCount);

			uint path_start_x = 0;
			if (mainPath)
			{
				if (isCoalMines)
				{
					path_start_x = 0x8f;
				}
				else
				{
					path_start_x = fillMainPath(mapSegment);
				}
			}

			int x = path_start_x;

			if (!mainPath) {
				while (x < map_w)
				{
					long c = getPixelColor(mapSegment, map_w, x, 0);
					if (c != COLOR_BLACK00)
					{
						x++;
						continue;
					}
				}
			}
			//printf("--#%i x:%i\n", idx, x);
			dSearch[threadIdx.x].map = mapSegment;
			dSearch[threadIdx.x].visited = visitedSegment;
			dSearch[threadIdx.x].queueMem = queueMemSegment;
			dSearch[threadIdx.x].queueSize = 1;
			dSearch[threadIdx.x].threadIdx = threadIdx.x;
			dSearch[threadIdx.x].targetX = x;
			dSearch[threadIdx.x].targetY = map_h - 1;
			dSearch[threadIdx.x].pathFound = 0;

			dSearch[threadIdx.x].visited[x] = 1;
			setPixelColor(dSearch[threadIdx.x].map, map_w, x, 0, COLOR_PURPLE);
			queueMemSegment[0] = { x, 0 };

			stackSize[threadIdx.x] = 0;
			stackCache[threadIdx.x * 4] = { -1,-1 };
			stackCache[threadIdx.x * 4 + 1] = { -1,-1 };
			stackCache[threadIdx.x * 4 + 2] = { -1,-1 };
			stackCache[threadIdx.x * 4 + 3] = { -1,-1 };
			
			dSearch[threadIdx.x].findPath(path_start_x, 0);
			
			validBlock[idx] = dSearch[threadIdx.x].pathFound;
		}
	}
}

__global__ 
void blockCoalMineHax(
	byte* block, 
	byte* validBlock,
	int phase,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* segment = block + (idx * (3 * map_w * map_h));
			doCoalMineHax(segment, phase, map_w, map_h);
		}
	}
}

__global__
void blockInitRNG(
	NollaPrng* rngBlock1,
	NollaPrng* rngBlock2)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		uint worldSeed = worldSeedStart + idx;

		rngBlock1[idx] = GetRNG(map_w, worldSeed);
		rngBlock2[idx] = NollaPrng(0);
	}
}

__global__
void buildTS(
	byte* data,
	int tiles_w,
	int tiles_h)
{
	stbhw_build_tileset_from_image(data, tiles_w * 3, tiles_w, tiles_h);
}
__global__
void freeTS()
{
	stbhw_free_tileset();
}

__global__
void blockGenerateMap(
	byte* resBlock,
	NollaPrng* rngBlock1,
	NollaPrng* rngBlock2,
	byte* validBlock,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* res = resBlock + idx * (3 * map_w * (map_h + 4));

			rngBlock2[idx].Seed = rngBlock1[idx].Next() * 2147483645.0;
			rngBlock2[idx].Next();

			stbhw_generate_image(res, map_w * 3, map_w, map_h + 4, &StaticRandom, rngBlock2 + idx);
		}
	}
}

__global__
void blockMemcpyOffset(
	byte* fromBlock,
	byte* toBlock,
	byte* validBlock,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* from = fromBlock + idx * (3 * map_w * (map_h + 4));
			byte* to = toBlock + idx * (3 * map_w * map_h);

			memcpy(to, from, 3 * map_w * map_h);
		}
	}
}

//prepare your eyes for some of the most horrific pointer code ever created
__global__ void blockCheckSpawnables(
	byte* mapBlock,
	byte* retArray,
	byte* validBlock,
	byte greedCurse) {
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (validBlock[idx]) {
			uint worldSeed = worldSeedStart + idx;
			byte* map = mapBlock + idx * (3 * map_w * map_h);
			byte* retSegment = retArray + idx * ((9 + maxChestContents) * (2 * pwCount + 1) * maxChestsPerWorld + sizeof(uint)) + sizeof(uint);
			int chestIdx = 0;
			bool densityExceeded = false;
			for (int px = 0; px < map_w; px++)
			{
				for (int py = 0; py < map_h; py++)
				{
					int pixelPos = 3 * (px + py * map_w);

					int gpX = GetGlobalPosX(worldX, worldY, px * 10, py * 10);
					int gpY = GetGlobalPosY(worldX, worldY, px * 10, py * 10 - 40);

					int PWSize = (ngPlus > 0 ? 64 : 70) * 512;

					if (map[pixelPos] == 0x78 && map[pixelPos + 1] == 0xFF && map[pixelPos + 2] == 0xFF) {
						for (int i = -pwCount; i <= pwCount; i++) {
							if (chestIdx >= (2 * pwCount + 1) * maxChestsPerWorld) {
								printf("Chest density exceeded in seed %i!\n", worldSeed);
								densityExceeded = true;
								break;
							}

							byte* c = retSegment + chestIdx * (9 + maxChestContents);
							spawnHeart(gpX + PWSize * i, gpY, worldSeed, c);
							if (loggingLevel >= 6) printf("Chest (%i %i) -> %i %i: %i\n", gpX, gpY, readUnalignedInt(c), readUnalignedInt(c + 4), *(c + 8));
							if (readUnalignedInt(c) != -1)
								chestIdx++;
						}
					}
					else if (map[pixelPos] == 0x55 && map[pixelPos + 1] == 0xff && map[pixelPos + 2] == 0x8c) {
						for (int i = -pwCount; i <= pwCount; i++) {
							if (chestIdx >= (2 * pwCount + 1) * maxChestsPerWorld) {
								printf("Chest density exceeded in seed %i!\n", worldSeed);
								densityExceeded = true;
								break;
							}

							byte* c = retSegment + chestIdx * (9 + maxChestContents);
							spawnChest(gpX + PWSize * i, gpY, worldSeed, greedCurse, c);
							if(loggingLevel >= 6) printf("Chest (%i %i) -> %i %i: %i\n", gpX, gpY, readUnalignedInt(c), readUnalignedInt(c + 4), *(c + 8));
							chestIdx++;
						}
					}
					if (densityExceeded) break;
				}
				if (densityExceeded) break;
			}
			writeUnalignedInt(retSegment - 4, chestIdx);
		}
	}
}

extern "C" {
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
		uint _worldSeedCount,
		uint maxTries,
		uint _pwCount,
		byte _ngPlus,
		byte _loggingLevel,
		uint _maxChestContents,
		uint _maxChestsPerWorld,
		byte _greedCurse) 
	{
		checkCudaErrors(cudaMemcpyToSymbol(map_w, &_map_w, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(map_h, &_map_h, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(isCoalMines, &_isCoalMine, sizeof(bool)));
		checkCudaErrors(cudaMemcpyToSymbol(worldX, &_worldX, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(worldY, &_worldY, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(worldSeedStart, &_worldSeedStart, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(worldSeedCount, &_worldSeedCount, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(pwCount, &_pwCount, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(ngPlus, &_ngPlus, 1));
		checkCudaErrors(cudaMemcpyToSymbol(maxChestContents, &_maxChestContents, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(maxChestsPerWorld, &_maxChestsPerWorld, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(loggingLevel, &_loggingLevel, 1));

		byte* dTileData;
		checkCudaErrors(cudaMalloc((void**)&dTileData, 3 * tiles_w * tiles_h));
		checkCudaErrors(cudaMemcpy(dTileData, host_tileData, 3 * tiles_w * tiles_h, cudaMemcpyHostToDevice));
		buildTS<<<1, 1>>>(dTileData, tiles_w, tiles_h);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaFree(dTileData));


		byte* resultBlock = (byte*)malloc(3 * _map_w * _map_h * _worldSeedCount);
		byte* validBlock = (byte*)malloc(_worldSeedCount);

		NollaPrng* rngBlock1;
		NollaPrng* rngBlock2;
		byte* dResultBlock;
		byte* dResBlock;
		byte* dValidBlock;
		intPair* dStackMem;

		checkCudaErrors(cudaMalloc((void**)&rngBlock1, sizeof(NollaPrng) * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&rngBlock2, sizeof(NollaPrng) * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dResultBlock, 3 * _map_w * _map_h * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dResBlock, 3 * _map_w * (_map_h + 4) * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dValidBlock, _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dStackMem, sizeof(intPair) * _worldSeedCount * (_map_w + _map_h)));

		checkCudaErrors(cudaMemset(dValidBlock, 0, _worldSeedCount));

		blockInitRNG<<<NUMBLOCKS, BLOCKSIZE>>>(rngBlock1, rngBlock2);
		checkCudaErrors(cudaDeviceSynchronize());

		bool stop = false;

		long long int mapgenTime = 0;
		long long int miscTime = 0;
		long long int validateTime = 0;

		int tries = 0;
		while (!stop) {
			if (tries > maxTries) break;

			chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
			blockGenerateMap<<<NUMBLOCKS, BLOCKSIZE>>>(dResBlock, rngBlock1, rngBlock2, dValidBlock, tries == 0);
			checkCudaErrors(cudaDeviceSynchronize());
			blockMemcpyOffset<<<NUMBLOCKS, BLOCKSIZE>>>(dResBlock, dResultBlock, dValidBlock, tries == 0);
			checkCudaErrors(cudaDeviceSynchronize());
			
			chrono::steady_clock::time_point time2 = chrono::steady_clock::now();

			blockFillC0FFEE << <NUMBLOCKS, BLOCKSIZE >> > (dResultBlock, dValidBlock, dResBlock, dStackMem, tries == 0);
			checkCudaErrors(cudaDeviceSynchronize());

			if (_isCoalMine) {
				blockCoalMineHax << <NUMBLOCKS, BLOCKSIZE >> > (dResultBlock, dValidBlock, 1, tries == 0);
				checkCudaErrors(cudaDeviceSynchronize());
			}
			if (_worldY < 20 && _worldX > 32 && _worldX < 39) {
				blockRoomBlock<<<NUMBLOCKS, BLOCKSIZE>>>(dResultBlock, dValidBlock, tries == 0);
				checkCudaErrors(cudaDeviceSynchronize());
			}
			if (_isCoalMine) {
				blockCoalMineHax << <NUMBLOCKS, BLOCKSIZE >> > (dResultBlock, dValidBlock, 2, tries == 0);
				checkCudaErrors(cudaDeviceSynchronize());
			}

			chrono::steady_clock::time_point time3 = chrono::steady_clock::now();
			blockIsValid<<<NUMBLOCKS, BLOCKSIZE>>>(dResultBlock, dValidBlock, dResBlock, dStackMem, tries == 0);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(validBlock, dValidBlock, _worldSeedCount, cudaMemcpyDeviceToHost));

			chrono::steady_clock::time_point time4 = chrono::steady_clock::now();
			mapgenTime += chrono::duration_cast<chrono::milliseconds>(time2 - time1).count();
			miscTime += chrono::duration_cast<chrono::milliseconds>(time3 - time2).count();
			validateTime += chrono::duration_cast<chrono::milliseconds>(time4 - time3).count();

			checkCudaErrors(cudaMemcpy(resultBlock + 3 * _map_w * _map_h * tries, dResultBlock, 3 * _map_w * _map_h, cudaMemcpyDeviceToHost));

			tries++;
			stop = false;
			int numBad = 0;
			for (int j = 0; j < _worldSeedCount; j++) if (!validBlock[j]) { numBad++; }
			if (numBad > 0) stop = false;

			if(_loggingLevel >= 3) printf("Try %i: Maps invalid: %i\n", tries, numBad);
		}
		checkCudaErrors(cudaFree(rngBlock1));
		checkCudaErrors(cudaFree(rngBlock2));
		checkCudaErrors(cudaFree(dResBlock));
		checkCudaErrors(cudaFree(dStackMem));
		freeTS << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());
		free(validBlock);

		byte* retArray = (byte*)malloc(_worldSeedCount * ((9 + _maxChestContents) * (2 * _pwCount + 1) * _maxChestsPerWorld + sizeof(uint)));
		byte* dRetArray;
		checkCudaErrors(cudaMalloc((void**)&dRetArray, _worldSeedCount * ((9 + _maxChestContents) * (2 * _pwCount + 1) * _maxChestsPerWorld + sizeof(uint))));

		blockCheckSpawnables<<<NUMBLOCKS, BLOCKSIZE >>>(dResultBlock, dRetArray, dValidBlock, _greedCurse);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(retArray, dRetArray, _worldSeedCount * ((9 + _maxChestContents) * (2 * _pwCount + 1) * _maxChestsPerWorld + sizeof(uint)), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(dResultBlock));
		checkCudaErrors(cudaFree(dValidBlock));
		checkCudaErrors(cudaFree(dRetArray));
		free(resultBlock);

		if (_loggingLevel >= 4) {
			printf("WORLDGEN ACCUMULATED TIME: %lli ms\n", mapgenTime);
			printf("VALIDATE ACCUMULATED TIME: %lli ms\n", validateTime);
			printf("MISCELL. ACCUMULATED TIME: %lli ms\n", miscTime);
		}

		return retArray;
	}
}

extern "C" {
	__declspec(dllexport) void free_array(void* block) {
		free(block);
	}
}

int main() {
	return 0;
}