#pragma once
#ifndef BVH_H
#define BVH_H

#include "LoadObj.hpp"
#include <algorithm>
#include <array>

struct Hit {
    public:
    const Triangle* triangle;
    glm::vec3 hitPoint;
    float t;
    Hit();

    Hit(const glm::vec3& p, const Triangle* t)
    : hitPoint(p), triangle(t) {}
};

class BVH {
    private:

    std::vector<Triangle>::iterator start, end;
    
    static constexpr int BINS = 12;
    static constexpr int nSplits = BINS - 1;

    // std::array<Bucket, BINS> buckets;
    // std::array<float, nSplits> costs;

    public:
    // std::vector<Triangle> triangles;
    BVH* left;
    BVH* right;
    glm::vec3 minVec;
    glm::vec3 maxVec;
    bool isLeaf = false;
        int count;

    static BVH* pool;
    static size_t poolPtr;
    static size_t poolSize;

    static BVH* alloc() {
        return &pool[poolPtr++];
    }        

    static void initPool(size_t nTriangles);

    BVH() {}

    BVH( std::vector<Triangle>::iterator start, 
         std::vector<Triangle>::iterator end );
    

    void aabb( std::vector<Triangle>::iterator start, std::vector<Triangle>::iterator end );
    void sortByAxis(int axis);
    void intersectTriangle(
        const glm::vec3& o,
        const glm::vec3& dir,
        const Triangle& tri,
        float tmin, float tmax,
        bool& isHit, float& t, float& u, float& v, float& w
    );
    void intersectAABB(
        const glm::vec3& o,
        const glm::vec3& r,
        bool& isHit, float& t_close, float& t_far
    );

    float IntersectAABB( const glm::vec3& o,
                         const glm::vec3& r,
                         float t, glm::vec3 bmin, glm::vec3 bmax );


    bool IntersectBVH( const glm::vec3& o, const glm::vec3& r, Hit& outHit, float tmax = 1e30f );

    // Hit *traverse(const glm::vec3& o, const glm::vec3& dir, float *tmax);
};

#endif