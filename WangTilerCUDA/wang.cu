#include <memory>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "noita_random.h"
#include "stb_hbwang.h"
#include "coalhax.h"
#include "spells.h"
#include "wand_levels.h"

//TODO fiddle with these to maximize performance, not sure what the correct configuration is
#define NUMBLOCKS 512
#define BLOCKSIZE 64

typedef unsigned char byte;

#define COLOR_PURPLE 0x7f007f
#define COLOR_BLACK00 0x000000
#define COLOR_BLACK01 0x010101
#define COLOR_BLACK02 0x020202
#define COLOR_WHITE 0xffffff
#define COLOR_YELLOW 0xffff00
#define COLOR_COFFEE 0xc0ffee

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
__device__ __constant__ byte biomeIndex = 0;

// pudy248 note: If more generation differences occur, this would be the place to start debugging.
#define BCSize 9
__device__ __constant__ unsigned long blockedColors[BCSize] = {
	0x00ac6e, //load_pixel_scene4_alt
	0x70d79e, //load_gunpowderpool_01
	0x70d79f, //???
	0x70d7a1, //load_gunpowderpool_04
	0x7868ff, //load_gunpowderpool_02
	0xc35700, //load_oiltank
	0xff0080, //load_pixel_scene2
	0xff00ff, //???
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
	rng.SetRandomFromWorldSeed(world_seed);
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
	NollaPrng rng = NollaPrng();
	rng.SetRandomFromWorldSeed(world_seed);
	rng.Next();
	int length = (int)((unsigned long long)((long long)map_w * -0x2e8ba2e9) >> 0x20);
	int iters = ((length >> 1) - (length >> 0x1f)) * 0xb + ((uint)world_seed / 0xc) * -0xc + world_seed + map_w;
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
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			long o = getPos(256, 3, x, y);
			long i = getPos(width, 3, x, y);
			long pix = createRGB(coalmine_overlay[o], coalmine_overlay[o + 1], coalmine_overlay[o + 2]);
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
			__syncthreads();
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
	int increment = dTileSet.short_side_len;
	for (int y = 1; y < height - 1; y+=increment)
	{
		for (int x = 1; x < width; x+=increment)
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

//__device__ __shared__ intPair stackCache[BLOCKSIZE * 4];
//__device__ __shared__ byte stackSize[BLOCKSIZE];

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

	intPair stackCache[4];
	byte stackSize;

	__device__
	void findPath(int x, int y)
	{
		int rmw = map_w; //register map width
		int rmh = map_h; //register map height
		while (queueSize > 0 && pathFound != 1)
		{
			stackSize = 0;
			intPair n = Pop();
			//if((n.x + n.y) % 2 == 0) setPixelColor(map, register_mapW, n.x, n.y, COLOR_PURPLE);
			if (n.x != -1) {
				if (atTarget(n))
				{
					pathFound = 1;
				}
				tryNext(n.x, n.y + 1, rmw, rmh);
				tryNext(n.x - 1, n.y, rmw, rmh);
				tryNext(n.x + 1, n.y, rmw, rmh);
				tryNext(n.x, n.y - 1, rmw, rmh);
			}
			Push();
		}
	}

	__device__ intPair Pop() {
		return queueMem[--queueSize];
	}
	
	__device__ void Push() {
		while (stackSize > 0) {
			queueMem[queueSize++] = stackCache[--stackSize];
		}
	}

	__device__
	void tryNext(int x, int y, int rmw, int rmh)
	{
		if (x >= 0 && y >= 0 && x < rmw && y < rmh) {
			if (visited[y * rmw + x] == 0 && traversable(x, y, rmw))
			{
				visited[y * rmw + x] = 1;
				stackCache[stackSize++] = { x, y };
			}
		}
	}

	__device__ bool traversable(int x, int y, int rmw)
	{
		long c = getPixelColor(map, rmw, x, y);

		return c == COLOR_BLACK00 || c == COLOR_COFFEE;
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

//why in god's name does the game store seed positions as 6 char strings???
__host__ __device__ int roundRNGPos(int num) {
	if (-1000000 < num && num < 1000000) return num;
	else if (-10000000 < num && num < 10000000) return num - (num % 10) + (num % 10 >= 5 ? 10 : 0);
	else if (-100000000 < num && num < 100000000) return num - (num % 100) + (num % 100 >= 50 ? 100 : 0);
	return num;
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
// 44 egg_purple
// 45 egg_slime
// 46 egg_monster
// 47 broken_wand
// 
// 64 wand_T10NS
// 65 wand_T1B
// 66 wand_T2B
// 67 wand_T3B
// 68 wand_T4B
// 69 wand_T5B
// 70 wand_T6B
// 
// random spell: 1xxxxxx0
// xxxxx = # of random calls to make
// 
// 253 sampo
// 255 orb

__device__ int MakeRandomCard(NoitaRandom* random) {
	int res = 0;
	char valid = 0;
	while (valid == 0) {
		int itemno = random->Random(0, 392);
		Spell item = all_spells[itemno];
		double sum = 0;
		for (int i = 0; i < 11; i++) sum += item.spawn_probabilities[i];
		if (sum > 0) {
			valid = 1;
			res = itemno;
		}
	}
	return res;
}

__device__ void CheckNormalChestLoot(int x, int y, uint worldSeed, byte expandSpells, byte* writeLoc)
{
	writeUnalignedInt(writeLoc, x);
	writeUnalignedInt(writeLoc + 4, y);
	byte* contents = writeLoc + 9;
	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(roundRNGPos(x)+509.7, y+683.1);

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
			else if (rnd <= 98) contents[idx++] = 4;
			else
			{
				rnd = random.Random(0, 100);
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
		else if (rnd <= 65) 
		{
			int amount = 1;
			int rnd2 = random.Random(0, 100);
			if (rnd2 <= 50) amount = 1;
			else if (rnd2 <= 70) amount += 1;
			else if (rnd2 <= 80) amount += 2;
			else if (rnd2 <= 90) amount += 3;
			else amount += 4;

			for (int i = 0; i < amount; i++) {
				random.Random(0, 1);
				if (expandSpells > 0) {
					int randCTR = random.randomCTR;
					contents[idx++] = (randCTR << 1) | 0x80;
				}
				MakeRandomCard(&random);
			}

			if(expandSpells == 0)
				contents[idx++] = 26;
		}
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
	random.SetRandomSeed(roundRNGPos(x), y);

	int idx = 0;
	int count = 1;

	if (random.Random(0, 100000) >= 100000)
	{
		count = 0;
		if (random.Random(0, 1000) == 999) contents[idx++] = 255;
		else contents[idx++] = 253;
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

__device__ void CheckItemPedestalLoot(int x, int y, uint worldSeed, byte* writeLoc) 
{
	writeUnalignedInt(writeLoc, x);
	writeUnalignedInt(writeLoc + 4, y);
	*(writeLoc + 8) = 1;
	byte* contents = writeLoc + 9;

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x + 425, y - 243);
	int rnd = random.Random(1, 91);

	if (rnd <= 65)
		contents[0] = 5;
	else if (rnd <= 70)
		contents[0] = 4;
	else if (rnd <= 71)
		contents[0] = 13;
	else if (rnd <= 72) {
		byte r_opts[7] = { 19, 20, 21, 22, 23, 24, 25 };
		rnd = random.Random(0, 6);
		byte r_opt = r_opts[rnd];
		contents[0] = r_opt;
	}
	else if (rnd <= 73)
		contents[0] = 44;
	else if (rnd <= 77)
		contents[0] = 45;
	else if (rnd <= 79)
		contents[0] = 46;
	else if (rnd <= 83)
		contents[0] = 16;
	else if (rnd <= 85)
		contents[0] = 15;
	else if (rnd <= 89)
		contents[0] = 47;
	else
		contents[0] = 14;
}

__device__ void spawnHeart(int x, int y, uint seed, byte greedCurse, byte expandSpells, byte* writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
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
				CheckNormalChestLoot(x, y, seed, expandSpells, writeLoc);
		}
	}
}

__device__ void spawnChest(int x, int y, uint seed, byte greedCurse, byte expandSpells, byte* writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
	if(loggingLevel >= 5) printf("Spawning guaranteed chest: %i, %i\n", x, y);
	random.SetRandomSeed(x, y);
	int super_chest_spawn_rate = greedCurse > 0 ? 100 : 2000;
	int rnd = random.Random(1, super_chest_spawn_rate);

	if (rnd >= super_chest_spawn_rate - 1)
		CheckGreatChestLoot(x, y, seed, writeLoc);
	else
		CheckNormalChestLoot(x, y, seed, expandSpells, writeLoc);
}

__device__ void spawnPotion(int x, int y, uint seed, byte greedCurse, byte expandSpells, byte* writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
	if (loggingLevel >= 5) printf("Spawning item pedestal: %i, %i\n", x, y);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
	{
		if (loggingLevel >= 5) printf("Spawning item on pedestal: %i, %i\n", x+5, y-4);
		CheckItemPedestalLoot(x + 5, y - 4, seed, writeLoc);
	}
}

__device__ void spawnPixelScene(int x, int y, uint seed, byte oiltank, byte greedCurse, byte expandSpells, byte* writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	if (loggingLevel >= 5) printf("Spawning pixel scene: %i, %i\n", x, y);
	int rnd = random.Random(1, 100);
	if (rnd <= 50 && oiltank == 0 || rnd > 50 && oiltank > 0) {
		float rnd2 = random.ProceduralRandomf(x, y, 0, 1) * 3;
		if (0.5f < rnd2 && rnd2 < 1) {
			spawnChest(x + 94, y + 224, seed, greedCurse, expandSpells, writeLoc);
		}
	}
}

__device__ void spawnPixelScene1(int x, int y, uint seed, byte greedCurse, byte expandSpells, byte* writeLoc) {
	spawnPixelScene(x, y, seed, 0, greedCurse, expandSpells, writeLoc);
}

__device__ void spawnOilTank(int x, int y, uint seed, byte greedCurse, byte expandSpells, byte* writeLoc) {
	spawnPixelScene(x, y, seed, 1, greedCurse, expandSpells, writeLoc);
}

__device__ void spawnWand(int x, int y, uint seed, byte greedCurse, byte expandSpells, byte* writeLoc) {
	NoitaRandom random = NoitaRandom(seed);
	float r = random.ProceduralRandomf(x, y, 0, 1);
	if (r < 0.47) return;
	r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
	if (r < 0.755) return;

	int nx = x - 5;
	int ny = y - 14;
	BiomeWands wandSet = wandLevels[biomeIndex];
	int sum = 0;
	for (int i = 0; i < wandSet.count; i++) sum += wandSet.levels[i].prob;
	r = random.ProceduralRandomf(nx, ny, 0, 1) * sum;
	for (int i = 0; i < wandSet.count; i++) {
		if (r <= wandSet.levels[i].prob) {
			writeUnalignedInt(writeLoc, nx+5);
			writeUnalignedInt(writeLoc + 4, ny+5);
			*(writeLoc + 8) = 1;
			*(writeLoc + 9) = wandSet.levels[i].id;
			return;
		}
		r -= wandSet.levels[i].prob;
	}
}

__device__ size_t sizeOfChest() {
	return 9 + maxChestContents;
}

__device__ size_t sizeOfChestSegment() {
	return sizeof(uint) + sizeOfChest() * maxChestsPerWorld * (2 * pwCount + 1);
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
	uint* seeds,
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
			uint worldSeed = seeds[idx];

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
					else break;
				}
			}

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

			dSearch[threadIdx.x].stackSize = 0;

			dSearch[threadIdx.x].findPath(path_start_x, 0);

			validBlock[idx] = dSearch[threadIdx.x].pathFound;
		}
	}
}

__global__ 
void blockCoalMineHax(
	byte* block, 
	byte* validBlock,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* segment = block + (idx * (3 * map_w * map_h));
			doCoalMineHax(segment, map_w, map_h);
		}
	}
}

__global__
void blockInitRNG(
	uint* seeds,
	NollaPrng* rngBlock1,
	NollaPrng* rngBlock2)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		uint worldSeed = seeds[idx];
		rngBlock1[idx] = GetRNG(map_w, worldSeed);
		rngBlock2[idx] = NollaPrng(0);
	}
}

__global__
void blockUpdateRNG(
	NollaPrng* rngBlock1,
	NollaPrng* rngBlock2)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		rngBlock2[idx].Seed = rngBlock1[idx].Next() * 2147483645.0;
		rngBlock2[idx].Next();
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
	NollaPrng* rngBlock,
	byte* validBlock,
	bool skipValid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		if (skipValid || !validBlock[idx]) {
			byte* res = resBlock + idx * (3 * map_w * (map_h + 4));

			stbhw_generate_image(res, map_w * 3, map_w, map_h + 4, &StaticRandom, rngBlock + idx);
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
	uint* seeds,
	byte* mapBlock,
	byte* retArray,
	byte* validBlock,
	byte greedCurse,
	byte checkItems,
	byte expandSpells,
	byte checkWands) {
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < worldSeedCount; idx += stride) {
		byte* retSegment = retArray + idx * sizeOfChestSegment();
		memset(retSegment, 0, sizeOfChestSegment());
		if (validBlock[idx]) {
			uint worldSeed = seeds[idx];
			byte* map = mapBlock + idx * (3 * map_w * map_h);
			int chestIdx = 0;
			bool densityExceeded = false;

			void (*spawnFuncs[6])(int, int, uint, byte, byte, byte*) = { spawnHeart, spawnChest, spawnPixelScene1, spawnOilTank, spawnPotion, spawnWand };

			for (int px = 0; px < map_w; px++)
			{
				for (int py = 0; py < map_h; py++)
				{
					int pixelPos = 3 * (px + py * map_w);

					int gpX = GetGlobalPosX(worldX, worldY, px * 10, py * 10);
					int gpY = GetGlobalPosY(worldX, worldY, px * 10, py * 10 - 40);

					int PWSize = (ngPlus > 0 ? 64 : 70) * 512;

					//avoids having to switch every loop
					auto func = spawnFuncs[0];
					long pix = createRGB(map[pixelPos], map[pixelPos + 1], map[pixelPos + 2]);

					switch (pix) {
					case 0x78ffff:
						func = spawnFuncs[0];
						break;
					case 0x55ff8c:
						func = spawnFuncs[1];
						break;
					case 0xff0aff:
						func = spawnFuncs[2];
						break;
					case 0xc35700:
						func = spawnFuncs[3];
						break;
					case 0x50a000:
						if (checkItems > 0)
							func = spawnFuncs[4];
						else continue;
						break;
					case 0x00ff00:
						if (checkWands > 0)
							func = spawnFuncs[5];
						else continue;
						break;
					default:
						continue;
					}

					for (int i = -pwCount; i <= pwCount; i++) {
						if (chestIdx >= (2 * pwCount + 1) * maxChestsPerWorld) {
							printf("Chest density exceeded in seed %i, with chest count above %i!\n", worldSeed, chestIdx);
							densityExceeded = true;
							break;
						}
						byte* c = retSegment + 4 + chestIdx * sizeOfChest();

						writeUnalignedInt(c, -1);
						writeUnalignedInt(c + 4, -1);
						*(c + 8) = 0;

						func(gpX + PWSize * i, gpY, worldSeed, greedCurse, expandSpells, c);

						if (readUnalignedInt(c) != -1) {
							if (loggingLevel >= 5) printf("Chest (%i %i) -> %i %i: %i\n", gpX, gpY, readUnalignedInt(c), readUnalignedInt(c + 4), *(c + 8));
							chestIdx++;
						}
					}
					if (densityExceeded) break;
				}
				if (densityExceeded) break;
			}
			writeUnalignedInt(retSegment, chestIdx);
		}
	}
}

__global__ void blockCheckEOE(
	int originX,
	int originY,
	uint radius,
	byte* retArray,
	byte checkItems,
	byte expandSpells,
	byte tinyMode) {
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
	for (int idx = index; idx < radius * radius * 4; idx += stride) {
		byte* c = retArray + idx * sizeOfChest();
		memset(c, 0, sizeOfChest());

		int x = originX - radius + (idx % (radius * 2));
		int y = originY - radius +  (idx / (radius * 2));

		if (tinyMode > 0) {
			writeUnalignedInt(c, x);
			writeUnalignedInt(c + 4, y);
			*(c + 8) = 1;
			*(c + 9) = 64;
		}
		else
			CheckGreatChestLoot(x, y, worldSeedStart, c);

		if (loggingLevel >= 5) printf("Chest (%i %i) -> %i %i: %i\n", x, y, readUnalignedInt(c), readUnalignedInt(c + 4), *(c + 8));
	}
}

extern "C" {
#ifdef _MSC_VER
	__declspec(dllexport)
#else
	__attribute__((visibility("default")))
#endif
	byte** generate_block(
		byte host_tileData[],
		uint seeds[],
		uint tiles_w,
		uint tiles_h,
		uint _map_w,
		uint _map_h,
		bool _isCoalMine,
		byte _biomeIndex,
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
		byte _greedCurse,
		byte _checkItems,
		byte _expandSpells,
		byte _checkWands)
	{
		checkCudaErrors(cudaMemcpyToSymbol(map_w, &_map_w, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(map_h, &_map_h, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(isCoalMines, &_isCoalMine, sizeof(bool)));
		checkCudaErrors(cudaMemcpyToSymbol(biomeIndex, &_biomeIndex, sizeof(byte)));
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

		uint* dSeeds;
		byte* resultBlock = (byte*)malloc(3 * _map_w * _map_h * _worldSeedCount);
		byte* validBlock = (byte*)malloc(_worldSeedCount);

		NollaPrng* rngBlock1;
		NollaPrng* rngBlock2;
		byte* dResultBlock;
		byte* dResBlock;
		byte* dValidBlock;
		intPair* dStackMem;

		checkCudaErrors(cudaMalloc((void**)&dSeeds, sizeof(uint) * _worldSeedCount));
		checkCudaErrors(cudaMemcpy(dSeeds, seeds, sizeof(uint) * _worldSeedCount, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void**)&rngBlock1, sizeof(NollaPrng) * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&rngBlock2, sizeof(NollaPrng) * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dResultBlock, 3 * _map_w * _map_h * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dResBlock, 3 * _map_w * (_map_h + 4) * _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dValidBlock, _worldSeedCount));
		checkCudaErrors(cudaMalloc((void**)&dStackMem, sizeof(intPair) * _worldSeedCount * (_map_w + _map_h)));

		checkCudaErrors(cudaMemset(dValidBlock, 0, _worldSeedCount));

		blockInitRNG<<<NUMBLOCKS, BLOCKSIZE>>>(dSeeds, rngBlock1, rngBlock2);
		checkCudaErrors(cudaDeviceSynchronize());

		bool stop = false;

		int tries = 0;
		if (_loggingLevel >= 5) printf("Beginning generation attempts.\n");
		while (!stop) {
			//chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
			if (tries >= maxTries) break;
			bool skipValid = tries == 0;

			blockUpdateRNG << <NUMBLOCKS, BLOCKSIZE >> > (rngBlock1, rngBlock2);
			checkCudaErrors(cudaDeviceSynchronize());

			blockGenerateMap<<<NUMBLOCKS, BLOCKSIZE>>>(dResBlock, rngBlock2, dValidBlock, skipValid);
			checkCudaErrors(cudaDeviceSynchronize());
			blockMemcpyOffset<<<NUMBLOCKS, BLOCKSIZE>>>(dResBlock, dResultBlock, dValidBlock, skipValid);
			checkCudaErrors(cudaDeviceSynchronize());

			if (_isCoalMine) {
				blockCoalMineHax << <NUMBLOCKS, BLOCKSIZE >> > (dResultBlock, dValidBlock, skipValid);
				checkCudaErrors(cudaDeviceSynchronize());
			}

			if (_worldY < 20 && _worldX > 32 && _worldX < 39) {
				blockRoomBlock<<<NUMBLOCKS, BLOCKSIZE>>>(dResultBlock, dValidBlock, skipValid);
				checkCudaErrors(cudaDeviceSynchronize());
			}

			checkCudaErrors(cudaMemset(dResBlock, 0, _worldSeedCount * _map_w * _map_h));
			blockIsValid<<<NUMBLOCKS, BLOCKSIZE>>>(dResultBlock, dValidBlock, dResBlock, dStackMem, skipValid);
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaMemcpy(validBlock, dValidBlock, _worldSeedCount, cudaMemcpyDeviceToHost));

			checkCudaErrors(cudaMemcpy(resultBlock, dResultBlock, 3 * _map_w * _map_h * _worldSeedCount, cudaMemcpyDeviceToHost));

			tries++;
			int numBad = 0;
			for (int j = 0; j < _worldSeedCount; j++) if (!validBlock[j]) { numBad++; }
			stop = numBad == 0;

			if(_loggingLevel >= 3) printf("Try %i: Maps invalid: %i\n", tries, numBad);
		}
		checkCudaErrors(cudaFree(rngBlock1));
		checkCudaErrors(cudaFree(rngBlock2));
		checkCudaErrors(cudaFree(dResBlock));
		checkCudaErrors(cudaFree(dStackMem));
		freeTS<<<1, 1>>>();
		checkCudaErrors(cudaDeviceSynchronize());
		free(validBlock);

		byte* retArray = (byte*)malloc(_worldSeedCount * (sizeof(uint) + (9 + _maxChestContents) * _maxChestsPerWorld * (2 * _pwCount + 1)));
		byte* dRetArray;
		checkCudaErrors(cudaMalloc((void**)&dRetArray, _worldSeedCount * (sizeof(uint) + (9 + _maxChestContents) * _maxChestsPerWorld * (2 * _pwCount + 1))));

		blockCheckSpawnables<<<NUMBLOCKS, BLOCKSIZE >>>(dSeeds, dResultBlock, dRetArray, dValidBlock, _greedCurse, _checkItems, _expandSpells, _checkWands);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(retArray, dRetArray, _worldSeedCount * (sizeof(uint) + (9 + _maxChestContents) * _maxChestsPerWorld * (2 * _pwCount + 1)), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(dResultBlock));
		checkCudaErrors(cudaFree(dValidBlock));
		checkCudaErrors(cudaFree(dRetArray));

		byte** retList = (byte**)malloc(sizeof(byte*) * 2);
		retList[0] = retArray;
		retList[1] = resultBlock;
		return retList;
	}
}

extern "C" {
#ifdef _MSC_VER
	__declspec(dllexport)
#else
	__attribute__((visibility("default")))
#endif
		byte* search_eoe(
			int originX,
			int originY,
			uint radius,
			uint _worldSeed,
			byte _loggingLevel,
			uint _maxChestContents,
			byte checkItems,
			byte expandSpells,
			byte tinyMode
		) {
		checkCudaErrors(cudaMemcpyToSymbol(worldSeedStart, &_worldSeed, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(maxChestContents, &_maxChestContents, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(loggingLevel, &_loggingLevel, 1));

		byte* retArray = (byte*)malloc(((9 + _maxChestContents) * 4 * radius * radius));
		byte* dRetArray;
		checkCudaErrors(cudaMalloc((void**)&dRetArray, (9 + _maxChestContents) * 4 * radius * radius));

		blockCheckEOE << <NUMBLOCKS, BLOCKSIZE >> > (originX, originY, radius, dRetArray, checkItems, expandSpells, tinyMode);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(retArray, dRetArray, (9 + _maxChestContents) * 4 * radius * radius, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(dRetArray));

		return retArray;
	}
}


//I don't trust freeing memory in C#, better to just P/Invoke the pointer back to C++ and free it there
extern "C" {
#ifdef _MSC_VER
	__declspec(dllexport)
#else
	__attribute__((visibility("default")))
#endif
		void free_array(void* block) {
		free(block);
	}
}

int main() {
	return 0;
}