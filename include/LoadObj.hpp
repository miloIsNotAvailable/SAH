#pragma once
#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include <cstddef>
#include "Triangle.hpp"
#include <string>
#include <vector>


extern std::vector<Triangle> triangles;
extern std::vector<Triangle> areaLights;
extern std::vector<tinyobj::material_t> Materials;

std::string loadFile();
void LoadObj();

#endif