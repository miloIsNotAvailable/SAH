#include "Triangle.hpp"

Triangle::Triangle(glm::vec3 i, glm::vec3 j, glm::vec3 k)
    : i(i), j(j), k(k)
{
    minVec = vecMin(minVec, vecMin(i, vecMin(j, k)));
    maxVec = vecMax(maxVec, vecMax(i, vecMax(j, k)));

    c = ( i + j + k ) / 3.f;

    e1 = j - i;
    e2 = k - i;

    n = glm::cross( e1, e2 );
}