#include <vector>
#include <algorithm>
#include "bvh.hpp"
#include <iterator>
#include <array>
#include <iostream>

Hit::Hit() {}

BVH* BVH::pool = nullptr;
size_t BVH::poolPtr = 0;
size_t BVH::poolSize = 0;

// constexpr int BVH::BINS = 12;
// constexpr int BVH::nSplits = BVH::BINS - 1;

void BVH::initPool( size_t nTriangles ) {
        poolSize = 2 * nTriangles;
        pool = new BVH[poolSize];
        poolPtr = 0;
}


struct AABB {
    
    public:
    glm::vec3 bmin, bmax;

    AABB() : 
    bmin( glm::vec3( std::numeric_limits<float>::infinity() ) ),
    bmax( glm::vec3( -std::numeric_limits<float>::infinity() ) )
    {}

    void grow( const glm::vec3& p ) {
        bmin.x = std::min( bmin.x, p.x );  
        bmin.y = std::min( bmin.y, p.y );  
        bmin.z = std::min( bmin.z, p.z );  
        
        bmax.x = std::max( bmax.x, p.x );  
        bmax.y = std::max( bmax.y, p.y );  
        bmax.z = std::max( bmax.z, p.z );  
    }

    float area() {
        glm::vec3 e = bmax - bmin;
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

struct Bucket {
    public:
    AABB aabb;
    int count;

    Bucket() : aabb( AABB() ), count( 0 ) {} 
    void reset() {
        count=0;
        aabb = AABB();
    }
};


BVH::BVH( std::vector<Triangle>::iterator start, std::vector<Triangle>::iterator end ) 
        : start(start), end(end) {

    minVec = glm::vec3( std::numeric_limits<float>::infinity() );
    maxVec = glm::vec3( -std::numeric_limits<float>::infinity() );

    count = std::distance(start, end);

    if (count <= 8) {
        isLeaf = true;
        this->start = start;
        this->end = end;
        return;
    }

    aabb( start, end );

    int bestAxis = -1;
    float bestPos = 0.f;
    float bestCost = std::numeric_limits<float>::infinity();
    float bestSplit = -1.f;
    
    glm::vec3 e = maxVec - minVec;
    float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
    float parentCost = count * parentArea;

    float bestMin, bestMax;

    // let leftTriangles = []
    // let rightTriangles = []
    
    auto leftStart = start;
    auto rightStart = start;
    auto leftEnd = end;
    auto rightEnd = end;

    // for (auto& b : buckets) b.reset();
    
    static constexpr int BINS = 12;
    static constexpr int nSplits = BINS - 1;
    // std::vector<int> axes = { 0, 1, 2 };
    for( int axis = 0; axis < 3; axis ++ ) {
        
        // std::array<Bucket, BINS> buckets;
        std::vector<Bucket> buckets( BINS );

        float boundsMin = std::numeric_limits<float>::infinity();
        float boundsMax = -std::numeric_limits<float>::infinity();
        
        for( auto i = start; i != end; i ++ ) {
            Triangle& tr = *i;
            
            boundsMin = std::min( boundsMin, tr.c[ axis ] );
            boundsMax = std::max( boundsMax, tr.c[ axis ] );
        } 
        
        // if( boundsMin == boundsMax ) continue;

        float scale = nSplits / (boundsMax - boundsMin + 1e-6);

        for( auto i = start; i != end; i ++ ) {
            Triangle& tr = *i;

            int cbms = (tr.c[axis] - boundsMin) * scale;
            int binIdx = std::min( nSplits, std::max( 0, cbms ) ); 
        
            buckets[binIdx].count ++;
            buckets[binIdx].aabb.grow( tr.i );
            buckets[binIdx].aabb.grow( tr.j );
            buckets[binIdx].aabb.grow( tr.k );
        }

        // std::cout << scale << std::endl;
        // std::cout << boundsMax << std::endl;

        AABB boundBelow = AABB();
        AABB boundAbove = AABB();
        int countBelow = 0;
        int countAbove = 0;

        std::vector<float> costs( nSplits, 0.f );
        // std::array<float, nSplits> costs;

        for( int i = 0; i < nSplits; i ++ ) {
            Bucket& b = buckets[ i ];
            
            if( b.count == 0 ) {
                costs[i]+= 0.; 
                continue;
            }

            countBelow += b.count;
            boundBelow.grow( b.aabb.bmin );
            boundBelow.grow( b.aabb.bmax );

            // std::cout << b.aabb.bmin.x << b.aabb.bmin.y << b.aabb.bmin.z << std::endl;
            // std::cout << b.count << std::endl;
            costs[i] += countBelow * boundBelow.area();
            // std::cout << countBelow * boundBelow.area() << std::endl;
        }

        for( int i = nSplits; i >= 1; i -- ) {
            Bucket& b = buckets[ i - 1 ];
            
            if( b.count == 0 ) {
                costs[i - 1]+= 0.; 
                continue;
            }

            countAbove += b.count;
            boundAbove.grow( b.aabb.bmin );
            boundAbove.grow( b.aabb.bmax );
            
            costs[i - 1] += countAbove * boundAbove.area();
        }

        // bestAxis = axis;
        for( int i = 0; i < nSplits; i ++ ) {
            // std::cout << "best cost: " << costs[i] <<  ", " << i << std::endl;
            if( costs[i] < bestCost && (costs[i] > 0) ) {
                bestAxis = axis;
                bestSplit = i+1;
                bestCost = costs[i];
                bestMin = boundsMin;
                bestMax = boundsMax;
            }
        }

        // for( auto &cost : costs ) {
        //     std::cout << cost << std::endl;
        // }
        // if( bestAxis == -1 ) 
        //     bestAxis = axis;
    }

    size_t leafCost = count;
    // bestCost =  1./2. + bestCost / parentArea;
    float c = 1.f/2.f + bestCost / parentArea;

    float splitPos = bestMin + (float(bestSplit) / float(BINS)) * (bestMax - bestMin);

    // std::cout << splitPos << ", " << count << ", " << bestAxis << ", " << c << ", " << leafCost << "\n";
    // std::cout << (count <= 8) << bestCost  << "\n";
    // std::cout << std::distance( start, end ) << ", " << float(count) << "\n";

    if( count <= 16 ) {
        // std::cout << "leaf of size: " << count << std::endl;
        isLeaf = true;
        this->start = start;
        this->end = end;
        return;

    }
    
    if( (leafCost <= c) || (count <= 16) ) {
        // std::cout << "leaf of size: " << count << std::endl;
        isLeaf = true;
        this->start = start;
        this->end = end;
        return;
    }

    if( count <= 2 ) {
        std::cout << "e" << std::endl;
    //   std::vector<Triangle>::iterator mid = std::next( start, count / 2 );
        int mid = count / 2;
        leftStart = start;
        leftEnd = start + mid;
      
        rightStart = start + mid;
        rightEnd = end;
      
    } else {      
      auto mid = std::partition( start, end, 
                                [ & ]( Triangle& t ) {
                                    return t.c[ bestAxis ] < splitPos;
                                } );

      leftStart = start;
      leftEnd = mid;

      rightStart = mid;
      rightEnd = end;
      
      // let 
    }

    left = BVH::alloc();
    right = BVH::alloc();

    *left = BVH( leftStart, leftEnd );
    *right = BVH( rightStart, rightEnd );

    // glm::vec3 extents = maxVec - minVec;
    
    // int longestAxisIndex = 0;
    // if (extents.y > extents.x) longestAxisIndex = 1;
    // if (extents.z > extents[longestAxisIndex]) longestAxisIndex = 2;        
    
    // sortByAxis( longestAxisIndex );
    
    // int mid = triangles.size() / 2.;
    // std::vector<Triangle> leftTris(triangles.begin(), triangles.begin() + mid);
    // std::vector<Triangle> rightTris(triangles.begin() + mid, triangles.end());
        
    // left = new BVH(leftTris, start, mid);
    // right = new BVH(rightTris, mid + 1, end);
    
}
    
void BVH::aabb( std::vector<Triangle>::iterator start, std::vector<Triangle>::iterator end ) {
    for (auto i = start; i != end; i ++) {
        Triangle& t = *i;
        for( auto& vert : { t.i, t.j, t.k } ) {
            minVec.x = std::min( minVec.x, vert.x );
            minVec.y = std::min( minVec.y, vert.y );
            minVec.z = std::min( minVec.z, vert.z );

            maxVec.x = std::max( maxVec.x, vert.x );
            maxVec.y = std::max( maxVec.y, vert.y );
            maxVec.z = std::max( maxVec.z, vert.z );
        }
        // minVec = vecMin(minVec, t.minVec);
        // maxVec = vecMax(maxVec, t.maxVec);
    }
}

void BVH::sortByAxis(int axis) {
    std::sort(triangles.begin(), triangles.end(),
        [&](Triangle& a, Triangle& b) {
            glm::vec3 ac = a.c;
            glm::vec3 bc = b.c;
            return ac[axis] < bc[axis];
        });
}
void BVH::intersectTriangle(
    const glm::vec3& o,
    const glm::vec3& dir,
    const Triangle& tri,
    float tmin, float tmax,
    bool& isHit, float& t, float& u, float& v, float& w
) {
    glm::vec3 e1 = tri.e1;
    glm::vec3 e2 = tri.e2 ;

    glm::vec3 h = glm::cross(dir, e2);
    float a = glm::dot(e1, h);
    if (std::fabs(a) < 1e-8f) {
        isHit = false; t = tmin; u = v = w = 0.f;
        return;
    }

    float f = 1.0f / a;
    glm::vec3 s = o - tri.i;
    u = f * glm::dot(s, h);
    if (u < 0.f || u > 1.f) {
        isHit = false; return;
    }

    glm::vec3 q = glm::cross(s, e1);
    v = f * glm::dot(dir, q);
    if (v < 0.f || u + v > 1.f) {
        isHit = false; return;
    }

    t = f * glm::dot(e2, q);
    if (t < tmin || t > tmax) {
        isHit = false; return;
    }

    isHit = true;
    w = 1.f - u - v;
}

float BVH::IntersectAABB( const glm::vec3& o,
                     const glm::vec3& r,
                     float t, glm::vec3 bmin, glm::vec3 bmax ) 
{
    float tx1 = (bmin.x - o.x) / r.x;
    float tx2 = (bmax.x - o.x) / r.x;

    float tmin = std::min( tx1, tx2 );
    float tmax = std::max( tx1, tx2 );

    float ty1 = (bmin.y - o.y) / r.y;
    float ty2 = (bmax.y - o.y) / r.y;

    tmin = std::max( tmin, std::min( ty1, ty2 ) );
    tmax = std::min( tmax, std::max( ty1, ty2 ) );

    float tz1 = (bmin.z - o.z) / r.z;
    float tz2 = (bmax.z - o.z) / r.z;

    tmin = std::max( tmin, std::min( tz1, tz2 ) );
    tmax = std::min( tmax, std::max( tz1, tz2 ) );

    if (tmax >= tmin && tmin < t && tmax > 0) return tmin; 
    else return 1e30;
}

bool BVH::IntersectBVH( const glm::vec3& o, const glm::vec3& r, Hit& outHit, float tmax ){
    int stackPtr = 0;
    // BVH* stack[ 64 ];
    // std::vector<BVH*> stack;
    std::array<BVH*, 128> stack;
    BVH* node = this;

    glm::vec3 hitPoint;
    float closestT = tmax;
    Triangle* hitTri = nullptr;

    while( true ) {
        
        if( node->isLeaf ) {
            for( auto i = node->start; i != node->end; i ++ ) {
                Triangle& tr = *i;
                bool isHit; float tHit, u, v, w;
                node->intersectTriangle(o, r, tr, 0.f, closestT, isHit, tHit, u, v, w);
                
                if( isHit && tHit < closestT ) {
                    closestT = tHit;
                    hitTri = &tr;
                    hitPoint = o + r * tHit;
                }
            }
            
            if( stackPtr == 0 ) break; 
            else { node = stack[--stackPtr]; }
            continue;
        }
        
        BVH* child1 = node->left;
        BVH* child2 = node->right;

        float distLeft = child1->IntersectAABB( o, r, closestT, child1->minVec, child1->maxVec );
        float distRight = child2->IntersectAABB( o, r, closestT, child2->minVec, child2->maxVec );

        if (distLeft <= distRight) {
            if (distRight < 1e30f) stack[stackPtr++] = child2;
            if (distLeft < 1e30f) node = child1;
            else if (stackPtr > 0) node = stack[--stackPtr];
            else break;
        } else {
            if (distLeft < 1e30f) stack[stackPtr++] = child1;
            if (distRight < 1e30f) node = child2;
            else if (stackPtr > 0) node = stack[--stackPtr];
            else break;
        }
    }

    if( !hitTri ) return false;

    outHit.hitPoint = hitPoint;
    outHit.triangle = hitTri;
    outHit.t = closestT;

    // Hit *hit = new Hit( hitPoint, *hitTri );
    // hit->t = closestT;    

    return true;
}

void BVH::intersectAABB(
    const glm::vec3& o,
    const glm::vec3& r,
    bool& isHit, float& t_close, float& t_far
) {
    glm::vec3 t_low  = (minVec - o) / r;
    glm::vec3 t_high = (maxVec - o) / r;

    glm::vec3 t_close_i(
        std::min(t_low.x, t_high.x),
        std::min(t_low.y, t_high.y),
        std::min(t_low.z, t_high.z)
    );
    glm::vec3 t_far_i(
        std::max(t_low.x, t_high.x),
        std::max(t_low.y, t_high.y),
        std::max(t_low.z, t_high.z)
    );

    t_close = std::max({t_close_i.x, t_close_i.y, t_close_i.z});
    t_far   = std::min({t_far_i.x,   t_far_i.y,   t_far_i.z});

    isHit = (t_close <= t_far);
}

// Hit *BVH::traverse(const glm::vec3& o, const glm::vec3& dir, float *tmax) {
//     // std::vector<BVH*> V;
//     // V.push_back(this);

//     BVH* V[ 64 ];
//     int idx = 0;
//     V[ idx++ ] = this;

//     float closestT = tmax ? *tmax : std::numeric_limits<float>::infinity();
//     glm::vec3 hitPoint(0.f);

//     Triangle* hitTri = nullptr;

//     while ( idx ) {
//         BVH* top = V[ --idx ];
//         // V.pop_back();

//         bool isect;
//         float t_close, t_far;
//         top->intersectAABB(o, dir, isect, t_close, t_far);
        
//         if (!isect || t_close > closestT) continue;

//         if (top->isLeaf) {
//             float closestHit = std::numeric_limits<float>::infinity();
//             for (auto& t : top->triangles) {
                
//                 bool isHit; float tHit, u, v, w;
//                 top->intersectTriangle(o, dir, t, 0.f, closestT, isHit, tHit, u, v, w);
                
//                 if (isHit && tHit < closestHit) {
//                     closestT = tHit;
//                     hitTri = &t;
//                     hitPoint = o + dir * tHit;
//                 }
//             }

//         } else {
//             bool isectLeft, isectRight;
//             float t_closeLeft, t_farLeft;
//             float t_closeRight, t_farRight;

//             if (top->left)
//                 top->left->intersectAABB(o, dir, isectLeft, t_closeLeft, t_farLeft);
//             else isectLeft = false;
//             if (top->right)
//                 top->right->intersectAABB(o, dir, isectRight, t_closeRight, t_farRight);
//             else isectRight = false;

//             if (isectLeft && isectRight) {
                
//                 BVH* nodeFrst = (t_closeLeft < t_closeRight) ? top->left : top->right;
//                 BVH* nodeScnd = (t_closeLeft < t_closeRight) ? top->right : top->left;
                
//                 // V.push_back(nodeFrst);
//                 // V.push_back(nodeScnd);

//                 V[ idx ++ ] = nodeFrst;
//                 V[ idx ++ ] = nodeScnd;

//             } else if (isectLeft) {
                
//                 // V.push_back(top->left);
//                 V[ idx ++ ] = top->left;

//             } else if (isectRight) {
                
//                 // V.push_back(top->right);
//                 V[ idx ++ ] = top->right;

//             }
//         }
//     }

//     if( !hitTri ) return nullptr;

//     Hit *hit = new Hit( hitPoint, *hitTri );
//     hit->t = closestT;
//     return hit;
// }