#include "random.hpp"

rng_type rng{ std::random_device{}() };
std::uniform_real_distribution<float> udist(0.f, 1.f);
