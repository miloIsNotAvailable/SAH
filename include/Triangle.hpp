#pragma once 

#ifndef TRIANGLE_H
#define TRIANGLE_H

// #define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <string>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <algorithm> 
#include <glm/glm.hpp>

inline glm::vec3 vecMin(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z));
}

inline glm::vec3 vecMax(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
}

class Triangle {
    public:
    glm::vec3 i, j, k;
    // tinyobj::material_t material;
    int materialID;
    std::string name;
    // aabb
    glm::vec3 minVec, maxVec;
    // edges
    glm::vec3 e1, e2;
    // centroid
    glm::vec3 c;
    // normal
    glm::vec3 n;

    Triangle() = default;
    Triangle( glm::vec3 i, glm::vec3 j, glm::vec3 k );

};

#endif