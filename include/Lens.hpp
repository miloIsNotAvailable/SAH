#pragma once

#ifndef LENS_H
#define LENS_H

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/glm.hpp>
#include "constants.hpp"
#include "random.hpp"

struct Lens {
    glm::vec3 point;
    glm::vec3 dir;

    Lens() = default;
    Lens( glm::vec3 point, glm::vec3 dir );
};

class Camera {
    public:
    float sensorDist, focalPlaneDist;
    float apertureSize;

    glm::vec3 eye, center, n;
    glm::vec3 sensor, focalPlane;

    glm::vec3 worldUp, right, up;

    Camera( float sensorDist, float focalPlaneDist, float apertureSize,
            glm::vec3& eye, glm::vec3& center );
    Lens thinLensRay( glm::vec2& st, float h );
};


#endif