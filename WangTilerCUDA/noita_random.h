#pragma once

#include <algorithm>
#include <math.h>
#include <vector>
// #include <limits.h>
#include <iostream>
#include <iterator>
#include <cfenv>
#include <cmath>


typedef unsigned int uint;
#ifdef _MSC_VER
typedef unsigned long long int ulong;
#endif

typedef uint8_t byte;
typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef uint8 bool8;

// Thanks to kaliuresis!
// Check out his orb atlas repository: https://github.com/kaliuresis/noa
// #include <stdint.h>



using namespace std;

class NollaPrng
{
public:
	double Seed;

	__host__ __device__
		NollaPrng(double seed)
	{
		Seed = seed;
		Next();
	}

	__host__ __device__
		uint H2(unsigned int a, unsigned int b, unsigned int ws)
	{
		unsigned int v3;
		unsigned int v4;
		unsigned int v5;
		int v6;
		unsigned int v7;
		unsigned int v8;
		int v9;

		v3 = (ws >> 13) ^ (b - a - ws);
		v4 = (v3 << 8) ^ (a - v3 - ws);
		v5 = (v4 >> 13) ^ (ws - v3 - v4);
		v6 = (v5 >> 12) ^ (v3 - v4 - v5);
		v7 = (v6 << 16) ^ (v4 - v6 - v5);
		v8 = (v7 >> 5) ^ (v5 - v6 - v7);
		v9 = (v8 >> 3) ^ (v6 - v7 - v8);
		return (((v9 << 10) ^ (v7 - v9 - v8)) >> 15) ^ (v8 - v9 - ((v9 << 10) ^ (v7 - v9 - v8)));
	}

	__host__ __device__
		void SetRandomFromWorldSeed(uint worldSeed)
	{
		Seed = worldSeed;
		if (2147483647.0 <= Seed)
		{
			Seed = worldSeed * 0.5;
		}
	}

	__host__ __device__
		void SetRandomSeed(uint ws, double x, double y)
	{
		uint a = ws ^ 0x93262e6f;
		uint b = a & 0xfff;
		uint c = (a >> 0xc) & 0xfff;

		double x_ = x + b;

		double y_ = y + c;

		double r = x_ * 134217727.0;
		uint64 e = SetRandomSeedHelper(r);

		uint64 _x = (*(uint64*)&x_ & 0x7fffffffffffffff);
		uint64 _y = (*(uint64*)&y_ & 0x7fffffffffffffff);
		if (102400.0 <= *((double*)&_y) || *((double*)&_x) <= 1.0)
		{
			r = y_ * 134217727.0;
		}
		else
		{
			double y__ = y_ * 3483.328;
			double t = e;
			y__ += t;
			y_ *= y__;
			r = y_;
		}

		uint64 f = SetRandomSeedHelper(r);

		uint g = SetRandomSeedHelper2(e, f, ws);
		double s = g;
		s /= 4294967295.0;
		s *= 2147483639.0;
		s += 1.0;

		if (2147483647.0 <= s)
		{
			s = s * 0.5;
		}

		Seed = s;

		Next();

		uint h = ws & 3;
		while (h)
		{
			Next();
			h--;
		}
	}

	NollaPrng() = default;

	__host__ __device__
		uint NextU()
	{
		Next();
		return (Seed * 4.656612875e-10) * 2147483645.0;
	}

	__host__ __device__
		double Next()
	{
		int v4 = (int)Seed * 0x41a7 + ((int)Seed / 0x1f31d) * -0x7fffffff;
		if (v4 < 0)
		{
			v4 += 0x7fffffff;
		}
		Seed = (double)v4;
		return Seed / 0x7fffffff;
	}

	__host__ __device__
		int Random(int a, int b)
	{
		return a + (int)((double)(b + 1 - a) * Next());
	}

	__host__ __device__
		float RoundHalfOfEven(float x)
	{
		std::fesetround(FE_TONEAREST);
		return std::nearbyint(x);
	}

	__host__ __device__
		uint64 SetRandomSeedHelper(double r)
	{
		uint64 e = *(uint64*)&r;
		if (((e >> 0x20 & 0x7fffffff) < 0x7ff00000) && (-9.223372036854776e+18 <= r) && (r < 9.223372036854776e+18))
		{
			// should be same as e &= ~(1<<63); which should also just clears the sign bit,
			// or maybe it does nothing,
			// but want to keep it as close to the assembly as possible for now
			e <<= 1;
			e >>= 1;
			double s = *(double*)&e;
			uint64 i = 0;
			if (s != 0.0)
			{
				uint64 f = (((uint64)e) & 0xfffffffffffff) | 0x0010000000000000;
				uint64 g = 0x433 - ((uint64)e >> 0x34);
				uint64 h = f >> g;

				int j = ~(uint)(0x433 < ((e >> 0x20) & 0xffffffff) >> 0x14) + 1;
				i = (uint64)j << 0x20 | j;
				i = ~i & h | f << (((uint64)s >> 0x34) - 0x433) & i;
				i = ~(~(uint64)(r == s) & (~i + 1) | i & (~(uint64)(r == s) + 1) + 1);
				// error handling, whatever
				// f = f ^
				// if((int) g > 0 && f )
			}
			return i & 0xffffffff;
		}

		// error!
		uint64 error_ret_val = 0x8000000000000000;
		return *(double*)&error_ret_val;
	}

	__host__ __device__
		uint SetRandomSeedHelper2(const uint a, const uint b, const uint ws)
	{
		uint uVar1;
		uint uVar2;
		uint uVar3;

		uVar2 = (a - b) - ws ^ ws >> 0xd;
		uVar1 = (b - uVar2) - ws ^ uVar2 << 8;
		uVar3 = (ws - uVar2) - uVar1 ^ uVar1 >> 0xd;
		uVar2 = (uVar2 - uVar1) - uVar3 ^ uVar3 >> 0xc;
		uVar1 = (uVar1 - uVar2) - uVar3 ^ uVar2 << 0x10;
		uVar3 = (uVar3 - uVar2) - uVar1 ^ uVar1 >> 5;
		uVar2 = (uVar2 - uVar1) - uVar3 ^ uVar3 >> 3;
		uVar1 = (uVar1 - uVar2) - uVar3 ^ uVar2 << 10;
		return (uVar3 - uVar2) - uVar1 ^ uVar1 >> 0xf;
	}
};

__device__ uint StaticRandom(NollaPrng* prng) {
	return prng->NextU();
}


class NoitaRandom
{
public:
	int randomCTR = 0;

	__host__ __device__
	NoitaRandom(uint worldSeed)
	{
		SetWorldSeed(worldSeed);
	}
	
	uint world_seed = 0;

	__host__ __device__
	ulong SetRandomSeedHelper(double r)
	{
		ulong e = *(ulong*)&r;

		if (((e >> 0x20 & 0x7fffffff) < 0x7ff00000) && (-9.223372036854776e+18 <= r) && (r < 9.223372036854776e+18))
		{
			e <<= 1;
			e >>= 1;
			double s = *(double*)&e;
			ulong i = 0;
			if (s != 0.0)
			{
				ulong f = (e & 0xfffffffffffff) | 0x0010000000000000;
				ulong g = 0x433 - (e >> 0x34);
				ulong h = f >> (int)g;

				uint j = ~(uint)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
				i = (ulong)j << 0x20 | j;
				i = ~i & h | f << (((int)s >> 0x34) - 0x433) & i;
				i = ~(~(uint)(r == s ? 1 : 0) + 1) & (~i + 1) | i & (~(uint)(r == s ? 1 : 0) + 1);
			}
			return i & 0xffffffff;
		}
	}

	__host__ __device__
	uint SetRandomSeedHelper2(uint a, uint b, uint ws)
	{
		uint uVar1;
		uint uVar2;
		uint uVar3;

		uVar2 = (a - b) - ws ^ ws >> 0xd;
		uVar1 = (b - uVar2) - ws ^ uVar2 << 8;
		uVar3 = (ws - uVar2) - uVar1 ^ uVar1 >> 0xd;
		uVar2 = (uVar2 - uVar1) - uVar3 ^ uVar3 >> 0xc;
		uVar1 = (uVar1 - uVar2) - uVar3 ^ uVar2 << 0x10;
		uVar3 = (uVar3 - uVar2) - uVar1 ^ uVar1 >> 5;
		uVar2 = (uVar2 - uVar1) - uVar3 ^ uVar3 >> 3;
		uVar1 = (uVar1 - uVar2) - uVar3 ^ uVar2 << 10;
		return (uVar3 - uVar2) - uVar1 ^ uVar1 >> 0xf;
	}

	double Seed;

	uint H2(uint a, uint b, uint ws)
	{
		uint v3;
		uint v4;
		uint v5;
		int v6;
		uint v7;
		uint v8;
		int v9;

		v3 = (ws >> 13) ^ (b - a - ws);
		v4 = (v3 << 8) ^ (a - v3 - ws);
		v5 = (v4 >> 13) ^ (ws - v3 - v4);
		v6 = (int)((v5 >> 12) ^ (v3 - v4 - v5));
		v7 = (uint)(v6 << 16) ^ (uint)(v4 - v6 - v5);
		v8 = (v7 >> 5) ^ (uint)(v5 - v6 - v7);
		v9 = (int)((v8 >> 3) ^ (uint)(v6 - v7 - v8));
		return (((uint)(v9 << 10) ^ (uint)(v7 - v9 - v8)) >> 15) ^ (uint)(v8 - v9 - ((uint)(v9 << 10) ^ (uint)(v7 - v9 - v8)));
	}

	__host__ __device__
	void SetRandomFromWorldSeed()
	{
		Seed = world_seed;
		if (2147483647.0 <= Seed)
		{
			Seed = world_seed * 0.5;
		}
	}

	__host__ __device__
	void SetRandomSeed(double x, double y)
	{
		uint ws = world_seed;
		uint a = ws ^ 0x93262e6f;
		uint b = a & 0xfff;
		uint c = (a >> 0xc) & 0xfff;

		double x_ = x + b;

		double y_ = y + c;

		double r = x_ * 134217727.0;
		ulong e = SetRandomSeedHelper(r);

		ulong _x = *(ulong*)&x_ & 0x7fffffffffffffff;
		ulong _y = *(ulong*)&y_ & 0x7fffffffffffffff;
		if (102400.0 <= *(double*)&_y || *(double*)&_x <= 1.0)
		{
			r = y_ * 134217727.0;
		}
		else
		{
			double y__ = y_ * 3483.328;
			double t = e;
			y__ += t;
			y_ *= y__;
			r = y_;
		}

		ulong f = SetRandomSeedHelper(r);

		uint g = SetRandomSeedHelper2((uint)e, (uint)f, ws);
		double s = g;
		s /= 4294967295.0;
		s *= 2147483639.0;
		s += 1.0;

		if (2147483647.0 <= s)
		{
			s *= 0.5;
		}

		Seed = s;

		Next();

		uint h = ws & 3;
		while (h > 0)
		{
			Next();
			h--;
		}
	}

	__host__ __device__
	uint NextU()
	{
		Next();
		return (uint)((Seed * 4.656612875e-10) * 2147483645.0);
	}

	__host__ __device__
	double Next()
	{
		int v4 = (int)Seed * 0x41a7 + ((int)Seed / 0x1f31d) * -0x7fffffff;
		if (v4 < 0)
		{
			v4 += 0x7fffffff;
		}
		Seed = v4;
		return Seed / 0x7fffffff;
	}

	__host__ __device__
	int Random(int a, int b)
	{
		randomCTR++;
		return a + (int)((b + 1 - a) * Next());
	}

	__host__ __device__
	void SetWorldSeed(uint worldseed)
	{
		world_seed = worldseed;
	}

	__host__ __device__
	float ProceduralRandomf(double x, double y, double a, double b)
	{
		SetRandomSeed(x, y);
		return (float)(a + ((b - a) * Next()));
	}

	__host__ __device__
	int ProceduralRandomi(double x, double y, double a, double b)
	{
		SetRandomSeed(x, y);
		return Random((int)a, (int)b);
	}
};