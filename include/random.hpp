#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include <random>

using rng_type = std::mt19937;

extern rng_type rng;
extern std::uniform_real_distribution<float> udist;

inline float hash(float)
{
    return udist(rng);
}

#endif