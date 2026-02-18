// ts already in Triangle.hpp
#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <vector>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
// #include "Triangle.hpp"
#include "bvh.hpp"
#include "Lens.hpp"
// #include "random.hpp"
#include <future>
#include "stb_image_write.h"
#include "LoadObj.hpp"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

struct Pixel {
    glm::vec3 color=glm::vec3( 0. );
    glm::vec2 coord;

    Pixel( glm::vec2 coord ) : coord(coord) {}
    Pixel() : color( glm::vec3( 0. ) ) {}
};

const GLuint WIDTH = 400, HEIGHT = 400;

glm::vec3 center(-10.0f, 18.f, 400.0f);
glm::vec3 eye(-120.0f, -58.f, -100.0f);
Camera camera( 1.5f, 840.9f, 1.15f, eye, center );

const int SPP = 10;

glm::vec3 cosineDirection( float seed, float cosThetaMax, glm::vec3 w)
{
    // compute basis from normal
    // see http://orbit.dtu.dk/fedora/objects/orbit:113874/datastreams/file_75b66578-222e-4c7d-abdf-f7e255100209/content
//     vec3 tc = vec3( 1.0+nor.z-nor.xy*nor.xy, -nor.x*nor.y)/(1.0+nor.z);
//     vec3 uu = vec3( tc.x, tc.z, -nor.x );
//     vec3 vv = vec3( tc.z, tc.y, -nor.y );
    
//     float u = random( gl_FragCoord.xy/u_resolution.xy + hash(seed)  );
//     float v = random( gl_FragCoord.xy/u_resolution.xy + hash(seed)  );
//     float a = 6.283185 * v;

//     return sqrt(u)*(cos(a)*uu + sin(a)*vv) + sqrt(1.0-u)*nor;

    float u1 = hash( seed );
    float u2 = hash( seed );

    // float cosTheta = (1.0f - u1) + u1 * cosThetaMax;
    float cosTheta = 1.0f - u1 * cosThetaMax;

    float sinTheta = sqrt(std::max(0.f, 1 - cosTheta * cosTheta));
    float phi = 2 * PI * u2;

    glm::vec3 localDir = glm::vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );

    glm::vec3 up = (abs(w.y) < 0.999f)
        ? glm::vec3(0, 1, 0)
        : glm::vec3(1, 0, 0);

    glm::vec3 u = glm::normalize(glm::cross(up, w));
    glm::vec3 v = glm::cross(w, u);

    return localDir.x * u + localDir.y * v + localDir.z * w;
}

glm::vec3 RandomUnitVectorInHemisphereOf(const glm::vec3& normal, float sa) {
    float r1 = hash( sa );
    float r2 = hash( sa );
    
    // Convert to polar coordinates
    float phi = 2.0 * PI * r1;
    float cosTheta = sqrt(1.0 - r2);
    float sinTheta = sqrt(r2);
    
    // Calculate the direction in local space
    glm::vec3 localDir = glm::vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    
    // Create a transformation matrix from the normal
    glm::vec3 up = abs(normal.y) < 0.999 ? glm::vec3(0.0, 1.0, 0.0) : glm::vec3(1.0, 0.0, 0.0);
    glm::vec3 tangentX = normalize(cross(up, normal));
    glm::vec3 tangentY = cross(normal, tangentX);
    
    // Transform the local direction to world space
    glm::vec3 worldDir = localDir.x * tangentX + localDir.y * tangentY + localDir.z * normal;
    
    return normalize(worldDir);
}

std::string generateRandomString(int length)
{
    // Define the list of possible characters
    const std::string CHARACTERS
        = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv"
          "wxyz0123456789";

    // Create a random number generator
    std::random_device rd;
    std::mt19937 generator(rd());

    // Create a distribution to uniformly select from all
    // characters
    std::uniform_int_distribution<> distribution(
        0, CHARACTERS.size() - 1);

    // Generate the random string
    std::string random_string;
    for (int i = 0; i < length; ++i) {
        random_string
            += CHARACTERS[distribution(generator)];
    }

    return random_string;
}

// enum LightType { SUN, POINT, AREA };

// struct Light {

//     public:
//     glm::vec3 pos, Li, normal;
//     float I;
//     LightType type;
//     Light::Light( glm::vec3 pos, glm::vec3 Li, glm::vec3 normal, float I, float r ): pos(pos), Li(Li), I(I), normal(normal) {
//         type = LightType::AREA;
//     }

//     Light::Light( glm::vec3 pos, glm::vec3 Li, glm::vec3 normal, float I ): pos(pos), Li(Li), I(I), normal(normal) {
//         type = LightType::AREA;
//     }

// };

struct LightSample {
    public:
    glm::vec3 pos, dir, normal;
    float distance2;
    LightSample( glm::vec3 &pos, glm::vec3 &dir, glm::vec3 &normal, float distance2 ) :  
    pos(pos), normal(normal), dir(dir), distance2(distance2) {}
};

class Light {
    public:
    virtual ~Light() = default;
    virtual LightSample sample( glm::vec3 p, float sa ) = 0;
    virtual float pdf( float dist2, float cosThetaL ) = 0;
    virtual glm::vec3 Li( glm::vec3 p, LightSample sample, BVH *bvh, float cosThetaL, float pdf ) = 0;
};

class SphereAreaLight : public Light {
    public: 
    glm::vec3 pos, Le;
    float radius;
    SphereAreaLight( glm::vec3 pos, glm::vec3 Le, float radius ) 
    : pos(pos), Le(Le), radius(radius) {
    }

    LightSample sample( glm::vec3 p, float sa ) {
        float rndTheta = hash( sa ) * 2.f * float(PI);
        float rndZ = hash( sa ) * 2.f - 1.f;
        float rndX = sqrt( 1 - rndZ * rndZ ) * cos( rndTheta );
        float rndY = sqrt( 1 - rndZ * rndZ ) * sin( rndTheta );

        glm::vec3 rndPoint = pos + radius * glm::vec3( rndX, rndY, rndZ );
        glm::vec3 normal = glm::normalize( rndPoint - pos );

        glm::vec3 lDir = rndPoint - p;
        glm::vec3 liray = glm::normalize( lDir );

        return LightSample( pos, liray, normal, glm::dot(lDir, lDir) ); 
    }

    float pdf( float dist2, float cosThetaL ) {
        float pdf_area = 1.f / (4.f * PI * radius * radius);
        float pdf_omega = pdf_area * dist2 / cosThetaL;

        return pdf_omega;
    }

    glm::vec3 Li( glm::vec3 p, LightSample sample, BVH *bvh, float cosThetaL, float pdf ) {
        Hit pointHit;
        bool isLightHit = bvh->IntersectBVH(p, sample.dir, pointHit, sqrt( sample.distance2 ) - 1e-4f);

        float shadow = float(1.f - isLightHit);

        return Le * cosThetaL * shadow / pdf;
    }
};

class SunLight : public Light {
    public: 
    glm::vec3 sunDir, Le;
    float cosThetaMax;
    SunLight( glm::vec3 sunDir, glm::vec3 Le, float cosThetaMax ) 
    : sunDir(sunDir), Le(Le), cosThetaMax(cosThetaMax) {
    }

    LightSample sample( glm::vec3 p, float sa ) {
        glm::vec3 dir = -cosineDirection( sa, cosThetaMax, sunDir );
        glm::vec3 v(0.f);
        dir = glm::normalize( dir );
        return LightSample( v, dir, sunDir, 1e30f );
    }

    float pdf( float dist2, float cosThetaL ) {
        float area = 2.f * float(PI) * (1.f - cosThetaMax);

        return 1.f / area;
    }

    glm::vec3 Li( glm::vec3 p, LightSample sample, BVH *bvh, float cosThetaL, float pdf ) {
        Hit pointHit;
        bool isLightHit = bvh->IntersectBVH(p, sample.dir, pointHit, sqrt( sample.distance2 ) - 1e-4f);
        
        float shadow = float(1.f - isLightHit);
        // printf( "%f\n", Le.x * cosThetaL * shadow / pdf );

        return Le * cosThetaL * shadow / pdf;
    }
};

glm::vec3 Fresnel_Schlick(float cosTheta, glm::vec3 F0)
{
    // Clamp to avoid negative cosines
    // cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(1.0f - cosTheta, 5.f);
}

inline float PowerHeuristic( int nf, float fPdf, int ng, float gPdf ) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return ( f * f ) / ( f * f + g * g );
}

float SmithG1_Beckmann(glm::vec3 v, float alpha)
{
    float cosTheta = abs(v.z);

    if (cosTheta <= 1e-4f)
        return 0.0f;

    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float tanTheta = sinTheta / cosTheta;

    if (tanTheta == 0.0f)
        return 1.0f;

    float a = 1.0f / (alpha * tanTheta);

    if (a >= 1.6f)
        return 1.0f;

    float a2 = a * a;
    return (3.535f * a + 2.181f * a2) /
           (1.0f + 2.276f * a + 2.577f * a2);
}

float BeckmannNDF( glm::vec3 &wh, float a ) {
    float cosTheta = abs(wh.z);
    float sinTheta = sqrt( std::max(0.f, 1.f - cosTheta * cosTheta) );

    if( cosTheta <= 1e-4f ) return 0.;

    float tan2Theta = (sinTheta * sinTheta) / (cosTheta * cosTheta);
    float cos4Theta = cosTheta * cosTheta * cosTheta * cosTheta;
    float a2 = a * a;

    return exp( -tan2Theta / a2 )  / (float(PI) * a2 * cos4Theta);
}

float BeckmannPDF( glm::vec3 &wo, glm::vec3 &wi, float a ) {

    // return BeckmannNDF( wh, a ) * abs(wh.z);
    glm::vec3 wh = glm::normalize( wo + wi );
    float D = BeckmannNDF( wh, a );
    float cosTheta_o = abs( wo.z );
    float G1 = SmithG1_Beckmann( wo, a );

    return D * G1 * abs(glm::dot(wo, wh)) / (cosTheta_o);
}

glm::vec3 SampleBeckmannIsotropic( float a, float sa ) {
    float a2 = a*a;
    float phi = float(2.f * PI) * hash( sa );
    float logSample = log( 1.f - hash(sa) );
    float tan2Theta = -a2 * logSample;

    // if( isinf( logSample ) ) return 0.f;

    float cosTheta = 1.f/sqrt(1.f + tan2Theta);
    float sinTheta = sqrt( std::max(0.f, 1.f - cosTheta * cosTheta) );

    glm::vec3 wh = glm::vec3( sinTheta * cos( phi ), sinTheta * sin( phi ), cosTheta );

    return wh;
}

glm::vec3 BeckamnnBRDF( glm::vec3 wo, glm::vec3 wi, glm::vec3 F0, float a ) {
    
    float cosTheta_o = abs( wo.z );
    float cosTheta_i = abs( wi.z );
    // glm::vec3 w = glm::normalize(wo + wi);
    glm::vec3 wh = glm::normalize(wo + wi);

    if( cosTheta_i <= 1e-4f || cosTheta_o <= 1e-4f ) return glm::vec3(0.f);

    glm::vec3 F = Fresnel_Schlick( abs(glm::dot(wo, wh)), F0 );
    float D = BeckmannNDF( wh, a );
    float G = SmithG1_Beckmann( wo, a ) * SmithG1_Beckmann(wi, a );
    // float G1 = SmithG1_Beckmann( wo, a );
    glm::vec3 fr = D * F * G / (4.f * cosTheta_i * cosTheta_o);
    

    // printf( "%f, %f, %f, %f, %f\n", fr.x, D, F.x, G, (4.f * cosTheta_i * cosTheta_o) );

    return fr;
}

glm::vec3 rendererCalculateColor( glm::vec3& ro, glm::vec3& rd, BVH *bvh, float sa ) {
    
    glm::vec3 lightCol = glm::vec3(1.000, .8, 1.);
    // glm::vec3 lightCol = glm::vec3(1.000, .8, .6);
    // glm::vec3 lightPos = glm::vec3(280., 400., 10.);
    glm::vec3 lightPos = glm::vec3(-60.0f, 28.f, -60.0f);
    // glm::vec3 lightPos = glm::vec3(-60.0f, 100.f, 100.0f);
    // glm::vec3 lightPos = glm::vec3(280., 400., 250.);
    // glm::vec3 lightPos = glm::vec3(-60.0f, 198.f, 400.0f);
    // glm::vec3 sunPos = glm::vec3(0.f, 0.f, 1.);
    
    float cosThetaMax = sqrt(3.)/2.f;
    // glm::vec3 sunCol = glm::vec3(1, 0.396, 0.659);
    glm::vec3 sunCol = glm::vec3(1, 0.894, 0.671) * 1.f;
    glm::vec3 sunPos = glm::normalize(glm::vec3( 1., 1., 1. ));
    
    glm::vec3 beta = glm::vec3( 1., 1., 1. );
    glm::vec3 L = glm::vec3( 0. );

    const int bounces = 7;
    
    // Light sun = Light( sunPos, sunCol, LightType::SUN );
    // sun.cosThetaMax = 2.f * sa - 1.f;

    // std::vector<Light> lights = {
    //     Light( lightPos, lightCol * 65000.0f, LightType::POINT ),
    //     // Light( lightPos, glm::vec3( 0.98, 0.549, 0.129 ) * 7.8f, LightType::POINT ),
    //     // Light( glm::vec3( 280., 500., 50. ), glm::vec3( 0, 0.851, 0.416 ) * 3.8f, LightType::POINT ),
    //     // sun,
    //     // Light( glm::vec3(300., 500., 250.), sunCol * 210.1f, LightType::POINT ),
    // };

    // float nLights = float(lights.size());
    // std::uniform_int_distribution<> distrib(0, nLights - 1);

    // SphereAreaLight light1 = SphereAreaLight( lightPos, 19.f * glm::vec3( 1. ), 20.f );
    // SunLight light2 = SunLight( glm::vec3( -1.f, -1.f, 1.f ), .1f * glm::vec3( 1. ), cos(float(PI)/20.f) );

    glm::vec3 sunDir = glm::normalize( glm::vec3( 0.f, 1.f, 1.f ) );

    std::vector<std::unique_ptr<Light>> lights;
    lights.push_back( std::make_unique<SphereAreaLight>(lightPos, 19.f * glm::vec3( 1. ), 20.f) );
    lights.push_back( std::make_unique<SunLight>(sunDir, 19.f * glm::vec3( 1. ), cos(20.f * float(PI)/180.f) ) );

    int nLights = lights.size();

    for( int i = 0; i < bounces; i ++ ) {

        Hit hit;
        bool isHit = bvh->IntersectBVH(ro, rd, hit);

        if (!isHit) {
            // delete hit;
            
            L += beta * glm::vec3(0.808, 0.949, 1);

            // if( i == 0 )
            //     L += glm::vec3(0.808, 0.949, 1);
                            
            break;
        }

        glm::vec3& pos = hit.hitPoint;
        glm::vec3 nor = glm::normalize(hit.triangle->n);
        bool frontFace = glm::dot( rd, nor ) > 0;
    
        if( frontFace ) nor=-nor;

        int matId = hit.triangle->materialID;
        tinyobj::material_t& mat = Materials[matId];

        glm::vec3 color = glm::vec3(mat.diffuse[0], 
                                    mat.diffuse[1], 
                                    mat.diffuse[2]);
        glm::vec3 bsdf = color / float(PI);

        // float rndTheta = hash( sa ) * 2.f * float(PI);
        // float rndZ = hash( sa ) * 2.f - 1.f;
        // float rndX = sqrt( 1 - rndZ * rndZ ) * cos( rndTheta );
        // float rndY = sqrt( 1 - rndZ * rndZ ) * sin( rndTheta );

        // // float radius = 50.f;
        // float radius = 20.f;

        // glm::vec3 rndPoint = lightPos + radius * glm::vec3( rndX, rndY, rndZ );        

        // glm::vec3 point = rndPoint;
        // glm::vec3 lDir = point - (pos);
        // glm::vec3 liray = glm::normalize( lDir );

        // glm::vec3 lightNormal = glm::normalize( rndPoint - lightPos );
        
        // glm::vec3 lightNormal = glm::normalize( rndPoint );
        
        int lightInd = nLights * hash(sa);
        
        LightSample s = lights[ lightInd ]->sample( pos, sa );
        // printf( "%f, %f, %f\n", s.dir.x, s.dir.y, s.dir.z );

        // float cosThetaL = std::max(dot(lightNormal, -liray), 0.f);
        float cosThetaL = std::max(dot(s.normal, -s.dir), 0.f);
        
        // float cosThetaL = abs(dot(lightNormal, -liray));

        // float pdf_area = 1.f / (4.f * PI * light.radius * light.radius);
        // float pdf_omega = pdf_area * s.distance2 / cosThetaL;

        glm::vec3 liray = s.dir;

        if( cosThetaL <= 0.f ) continue;

        float pdf_omega = lights[ lightInd ]->pdf( s.distance2, cosThetaL ) / nLights;
        // printf( "%f, %f\n", cosThetaL, pdf_omega );

        // Hit pointHit;
        // bool isLightHit = bvh->IntersectBVH(pos + nor * 1e-4f, liray, pointHit, sqrt(s.distance2) - 1e-4f);

        // if(!isLightHit)
        //     printf( "%f, %f, %f, %f, %f, %f, %f, %f, %f\n", pointHit.t, pointHit.hitPoint.x, pointHit.hitPoint.y, pointHit.hitPoint.z, pos.x, pos.y, pos.z, glm::length(lDir), glm::length( pointHit.hitPoint - pos ));

        // float NdotL = std::max( glm::dot( nor, liray ), 0.f );
        // float I = 19.f;
        // float I = 10.f;
        // float I = 13.f;

        // float shadow = (1.f - float(isLightHit));
        
        glm::vec3 Li = lights[lightInd]->Li( pos + nor * 1e-4f, s, bvh, cosThetaL, pdf_omega );
        // glm::vec3 Li2 = shadow * cosThetaL * I * glm::vec3(1.f) / pdf_omega;

        // printf( "%f, %f, %f ,.., %f, %f, %f\n", Li.x, Li.y, Li.z, Li2.x, Li2.y, Li2.z );

        // glm::vec3 bounceDir = RandomUnitVectorInHemisphereOf( nor, hash(sa) );
        // float pdf = 1. / ( 2. * PI );  

        glm::vec3 wo = -rd;
        // glm::vec3 wm = glm::normalize(Sample_wm( wo, sa, a_x, a_y ));
        // // if( glm::dot( glm::vec3(0., 0., 1.), wm ) < 0 ) wm=-wm;
       
        // glm::vec3 up = abs(nor.y) < 0.999 ? glm::vec3(0.0, 1.0, 0.0) : glm::vec3(1.0, 0.0, 0.0);
        // glm::vec3 tangentX = normalize(cross(up, nor));
        // glm::vec3 tangentY = cross(nor, tangentX);

        // wm = tangentX * wm.x + tangentY * wm.y + wm.z * nor;
        // wm = glm::normalize( wm );

        // phong lobe
        // float n = 600.f;

        // float u1 = hash(sa);
        // float u2 = hash(sa);

        // float phi = 2.f * PI * u1;
        // float ct = pow(u2, 1.f / (n + 1.f));
        // float sinTheta = sqrt(1.f - ct * ct);

        // // local space (around R)
        // glm::vec3 localDir(
        //     cos(phi) * sinTheta,
        //     sin(phi) * sinTheta,
        //     ct
        // );

        // glm::vec3 up = abs(nor.y) < 0.999 ? glm::vec3(0.0, 1.0, 0.0) : glm::vec3(1.0, 0.0, 0.0);
        // glm::vec3 tangentX = normalize(cross(up, nor));
        // glm::vec3 tangentY = cross(nor, tangentX);

        // glm::vec3 R = glm::reflect( -wo, nor );
        // glm::vec3 wi = tangentX * localDir.x + tangentY * localDir.y + localDir.z * R;
        // wi = glm::normalize( wi );
        // glm::vec3 wi = tangentX * localDir.x + tangentY * localDir.y + localDir.z * R;
        
        // beckmann        
        float roughness = mat.roughness;
        glm::vec3 localDir = SampleBeckmannIsotropic( roughness, sa );

        glm::vec3 up = abs(nor.y) < 0.999 ? glm::vec3(0.0, 1.0, 0.0) : glm::vec3(1.0, 0.0, 0.0);
        glm::vec3 tangentX = normalize(cross(up, nor));
        glm::vec3 tangentY = cross(nor, tangentX);

        if (localDir.z * wo.z <= 0.f) localDir = -localDir;

        glm::vec3 wh = tangentX * localDir.x + tangentY * localDir.y + localDir.z * nor;
        wh = glm::normalize( wh );   

        glm::vec3 wi = reflect( -wo, wh );
        glm::vec3 F0( .04f );
        glm::vec3 fr = BeckamnnBRDF( wo, wi, F0, roughness ); 
        float wh_pdf = BeckmannPDF( wo, wi, roughness );
        
        float WhDotWo = abs(dot(wo, wh));

        float pdf_wi = wh_pdf / (4.f * WhDotWo);
        // printf( "%f, %f, %f, %f, %f\n", glm::length( glm::normalize(wo + wi) - wh ), fr.y, fr.z, pdf_wi, wh_pdf );

        // float RdotWi = std::max(glm::dot(wi, R), 0.f);

        float pdf;

        glm::vec3 F = Fresnel_Schlick(std::max(0.f, glm::dot( wo, glm::normalize(wo + wi) )), F0);
        float p_spec = F.x;

        // float cosTheta_i = abs( wi.z );
        // float cosTheta_o = abs( wo.z );

        // if( cosTheta_i <= 1e-4f || cosTheta_o < 1e-4f ) continue;

        glm::vec3 diffuse = bsdf;
        // glm::vec3 diffuse = 28.f / (23.f * float(PI)) * color * (1.f-F) * (1.f - pow(1.f - .5f * cosTheta_i, 5.f)) * (1.f - pow(1.f - .5f * cosTheta_o, 5.f));
        // glm::vec3 specular = F * glm::vec3( 1. ) * (n+2.f)/(2.f * float(PI)) * pow(RdotWi, n);    
        glm::vec3 specular = fr;    
        
        glm::vec3 brdf = (1.f - p_spec) * diffuse + p_spec * specular;
        // glm::vec3 brdf = diffuse + specular;

        
        glm::vec3 bounceDir;
        float cosThetaMax = sqrt(3.f)/2.f;
        // float cosThetaMax = cos( float(PI/10.) );
        // float pdf_1, pdf_2;
        if( hash(sa) < p_spec ) {
            
            // glm::vec3 up = abs(nor.y) < 0.999 ? glm::vec3(0.0, 1.0, 0.0) : glm::vec3(1.0, 0.0, 0.0);
            // glm::vec3 tangentX = normalize(cross(up, nor));
            // glm::vec3 tangentY = cross(nor, tangentX);

            // glm::vec3 wi = tangentX * R.x + tangentY * R.y + R.z * nor;
            
            bounceDir = wi;
            // pdf_1 = GGX_pdf( wo, wm, a_x, a_y );
        } else {
            bounceDir = RandomUnitVectorInHemisphereOf( nor, hash(sa) );
            // bounceDir
            // float cosTheta = std::max(glm::dot(nor, bounceDir), 0.f);
            // pdf_2 = cosTheta / float(PI);
        }

        // printf("%f\n", pdf);
        
        Hit areaHit;
        bool aHit = bvh->IntersectBVH(pos + nor * 1e-4f, bounceDir, areaHit);
        
        float cosTheta = std::max(glm::dot(nor, bounceDir), 0.f);
        // pdf = p_spec * GGX_pdf( wo, wm, a_x, a_y ) + (1.f - p_spec) * cosTheta / float(PI);
        // float pdf_spec = (n + 1.f) / (2.f * PI) * pow(RdotWi, n);

        pdf = p_spec * (pdf_wi) + (1.f - p_spec) * cosTheta / float(PI);

        if( cosTheta <= 0.f ) continue;
        // if( pdf <= 0.f ) continue;
        // if( brdf.x == 0 && brdf.y == 0 && brdf.z == 0 ) continue;

        // glm::vec3 nextL = L;
        L += PowerHeuristic( 1, pdf, 1, pdf_omega ) * brdf * cosTheta * (1.f - float(aHit)) / pdf;
        // L += PowerHeuristic( 1, pdf_omega, 1, pdf ) * bsdf * NdotL * I * cosThetaL * shadow / (pdf_area * glm::dot(lDir, lDir));
        // L += PowerHeuristic( 1, pdf_omega, 1, pdf ) * brdf * I * cosThetaL * shadow / (pdf_omega);
        L += PowerHeuristic( 1, pdf_omega, 1, pdf ) * brdf * Li;
        L *= beta;
        

        // printf( "%f, %f, %f\n", L.x, L.y, L.z );

        // nextL = L;

        // L += beta * bsdf * NdotL * I * atten * shadow;

        // printf( "%f, %f, %f\n", pdf_omega, atten, beta.x * shadow * bsdf.x * I * atten / pdf_omega );

        cosTheta = std::max(glm::dot(nor, bounceDir), 0.f);
        // float pdf = 1. / ( 2. * PI );  

        // if( cosTheta <= 0.f ) continue;

        // pdf = cosTheta / ( PI );  
        // pdf = cosTheta / ( PI );  

        beta *= brdf * cosTheta / pdf;

        if (i > 3) {
            float p = glm::clamp(
                std::max(beta.r, std::max(beta.g, beta.b)),
                0.05f,
                0.95f
            ); 

            if (hash(sa) > p)
                break;

            beta /= p;
        }

        rd = bounceDir;
        ro = pos + nor * 1e-4f;

        // delete hit;
    }   

    // col = col / (col + glm::vec3(1.0));
    // col = glm::pow(col, glm::vec3(1.0 / 2.2));

    return L;
}

void renderBlock( std::vector<Pixel>& image, 
                  BVH* bvh, 
                  Camera& camera, 
                  glm::vec3& eye, 
                  glm::vec3& center, 
                  int start, int end,
                  int row, int col, int SAMPLES ) {

    for (int p = start; p < end; p += row)
    {
        int x = int(p % WIDTH);
        int y = int(p / HEIGHT) * col;

        glm::vec2 fragCoord = glm::vec2(float(x), float(y));
        glm::vec2 st = fragCoord / glm::vec2(WIDTH, HEIGHT) - 0.5f;

        Pixel pixel(fragCoord);

        for (int i = 1; i < SAMPLES; i++)
        {
            Lens worldDir = camera.thinLensRay( st, float(i));
            // Hit* hit = bvh->traverse(worldDir.point, worldDir.dir, nullptr);

            // pixel.color += rendererCalculateColor( worldDir.point, worldDir.dir, bvh, hash( i ) );
            glm::vec3 currentColor = rendererCalculateColor( worldDir.point, worldDir.dir, bvh, hash( i ) );

            pixel.color = ( pixel.color * float(i - 1) + currentColor ) / float(i);
        }

        // pixel.color /= float(SAMPLES);
        int idx = y * WIDTH + x;
        image[idx] = pixel;
    }
}

void renderSceneToImage(std::vector<Pixel>& image, BVH* bvh, Camera& camera, glm::vec3& eye, glm::vec3& center)
{
    int GRID_SIZE = WIDTH * HEIGHT;
    int SAMPLES = SPP;
    int row = 1, col = 1;

    std::vector<std::future<void>> futures;

    int numThreads = std::thread::hardware_concurrency();
    // int numThreads = 1;
    // int chunkSize = GRID_SIZE / std::thread::hardware_concurrency();
    // cool trick, ceiling division
    int chunkSize = (GRID_SIZE + numThreads - 1) / numThreads;

    for (int i = 0; i < GRID_SIZE; i += chunkSize) {

        futures.push_back(std::async(std::launch::async,
                                    renderBlock,
                                    std::ref(image),    // vector by reference
                                    bvh,                // pointer is fine
                                    std::ref(camera),   // camera by reference
                                    std::ref(eye),      // vec3 by reference
                                    std::ref(center),   // vec3 by reference
                                    i, std::min(GRID_SIZE, i + chunkSize),   // ints by value
                                    row, col, SAMPLES));
    }

    for (auto& f : futures)
        f.get();

    // for (int p = 0; p < GRID_SIZE; p += row)
    // {
    //     int x = int(p % WIDTH);
    //     int y = int(p / HEIGHT) * col;

    //     glm::vec2 fragCoord = glm::vec2(float(x), float(y));
    //     glm::vec2 st = fragCoord / glm::vec2(WIDTH, HEIGHT) - 0.5f;

    //     Pixel pixel(fragCoord);

    //     for (int i = 0; i < SAMPLES; i++)
    //     {
    //         Lens worldDir = camera.thinLensRay(.15f, st, eye, center, float(i));
    //         // Hit* hit = bvh->traverse(worldDir.point, worldDir.dir, nullptr);

    //         pixel.color += rendererCalculateColor( worldDir.point, worldDir.dir, bvh, hash( i ) );
    //     }

    //     pixel.color /= float(SAMPLES);
    //     int idx = y * WIDTH + x;
    //     image[idx] = pixel;
    // }
}

void updateTexture(GLuint textureID, const std::vector<Pixel>& image)
{
    std::vector<float> texData(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
    {
        texData[i * 3 + 0] = image[i].color.r;
        texData[i * 3 + 1] = image[i].color.g;
        texData[i * 3 + 2] = image[i].color.b;
    }

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT, texData.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

// cornell
// glm::vec3 center(278.0f, 270.4f, -700.0f);
// glm::vec3 eye(278.0f, 260.4f, 100.0f);
// Camera camera( 1.5f, 740.9f, 0.15f, eye, center );

// 263.437, 197.837, 525.034
// -303.239, -313.618, -360

// bunny
// glm::vec3 center(-10.0f, -28.f, 400.0f);
// glm::vec3 eye(-120.0f, 18.f, -100.0f);
// Camera camera( 1.5f, 840.9f, 1.15f, eye, center );

// Room
// glm::vec3 center(-20.0f, 70.4f, -100.0f);
// glm::vec3 eye(-20.0f, 60.4f, -500.0f);
// Camera camera( 1.5f, 840.9f, 1.15f, eye, center );

bool cameraMoved = true; 

int main()
{
    LoadObj();

    BVH::initPool( triangles.size() );

    BVH* bvh = new BVH(triangles.begin(), triangles.end());
    
    std::cout << "built bvh" << std::endl;
    std::cout << bvh->maxVec.x << bvh->maxVec.y << bvh->maxVec.z << std::endl;
    std::cout << bvh->minVec.x << bvh->minVec.y << bvh->minVec.z << std::endl;
    // std::cout << bvh->count << std::endl;

    // 992.046
    Hit hit;
    bool isHit = bvh->IntersectBVH( center, glm::normalize( eye - center ), hit, 900);

    if( isHit ) {
        std::cout << hit.t << std::endl;
    } else {
        std::cout << "no hit" << std::endl;
    }

    // row and col spacing
    int row = 1;
    int col = 1;
    int GRID_SIZE = WIDTH * HEIGHT;
    int SAMPLES = 1;
    // int GRID_SIZE = 100;
    
    std::vector<Pixel> image( GRID_SIZE );
    
    renderSceneToImage( image, bvh, camera, eye, center );
    std::cout << "saving to png" << std::endl;

    std::vector<unsigned char> buffer(WIDTH * HEIGHT * 3);

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        const glm::vec3& c = image[i].color;
        buffer[i*3 + 0] = static_cast<unsigned char>(std::min(1.f, c.r) * 255.f);
        buffer[i*3 + 1] = static_cast<unsigned char>(std::min(1.f, c.g) * 255.f);
        buffer[i*3 + 2] = static_cast<unsigned char>(std::min(1.f, c.b) * 255.f);
    }

    // Save as PNG
    std::string filename = generateRandomString( 5 ) + ".png";
    stbi_write_png(filename.c_str(), WIDTH, HEIGHT, 3, buffer.data(), WIDTH*3);

//     glfwInit();

//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//     glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

//     GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);
//     glfwMakeContextCurrent(window);
//     if (window == NULL)
//     {
//         std::cout << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }

//     glfwSetKeyCallback(window, key_callback);
//     glfwSetScrollCallback(window, scroll_callback);

//     int version = gladLoadGL(glfwGetProcAddress);
//     if (version == 0)
//     {
//         std::cout << "Failed to initialize OpenGL context" << std::endl;
//         return -1;
//     }

//     std::cout << "Loaded OpenGL " << GLAD_VERSION_MAJOR(version) << "." << GLAD_VERSION_MINOR(version) << std::endl;

//     GLuint textureID;
//     glGenTextures(1, &textureID);
//     glBindTexture(GL_TEXTURE_2D, textureID);

//     // Allocate the texture memory
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, nullptr);

//     // Set filtering
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

//     glBindTexture(GL_TEXTURE_2D, 0);

//     updateTexture(textureID, image);

//     std::string frag = R"(#version 330 core
// in vec2 TexCoords;
// out vec4 FragColor;

// uniform sampler2D screenTexture;

// void main() {
//     FragColor = texture(screenTexture, TexCoords);
// })";

//     std::string vert = R"(#version 330 core
// layout(location = 0) in vec2 aPos;
// layout(location = 1) in vec2 aTexCoords;

// out vec2 TexCoords;

// void main()
// {
//     TexCoords = aTexCoords;
//     gl_Position = vec4(aPos.xy, 0.0, 1.0);
// })";

//     const char* vertSrc = vert.c_str();
//     const char* fragSrc = frag.c_str();

//     GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
//     glShaderSource(vertexShader, 1, &vertSrc, NULL);
//     glCompileShader(vertexShader);

//     GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
//     glShaderSource(fragmentShader, 1, &fragSrc, NULL);
//     glCompileShader(fragmentShader);

//     GLuint shaderProgram = glCreateProgram();
//     glAttachShader(shaderProgram, vertexShader);
//     glAttachShader(shaderProgram, fragmentShader);
//     glLinkProgram(shaderProgram);

//     glDeleteShader(vertexShader);
//     glDeleteShader(fragmentShader);


//     float quadVertices[] = {
//         // positions    // texcoords
//         -1.0f, -1.0f,   0.0f, 0.0f,
//         1.0f, -1.0f,   1.0f, 0.0f,
//         1.0f,  1.0f,   1.0f, 1.0f,
//         -1.0f,  1.0f,   0.0f, 1.0f
//     };

//     unsigned int quadIndices[] = {
//         0, 1, 2,
//         2, 3, 0
//     };


//     unsigned int VAO, VBO, EBO;
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glGenBuffers(1, &EBO);

//     glBindVertexArray(VAO);

//     // Bind and fill VBO
//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

//     // Bind and fill EBO
//     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//     glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

//     // Vertex positions
//     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);

//     // Texture coordinates
//     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
//     glEnableVertexAttribArray(1);

//     // Unbind VAO (optional)
//     glBindVertexArray(0);

//     while (!glfwWindowShouldClose(window))
//     {
//         glfwPollEvents();

//         // if (cameraMoved)
//         // {
//         //     // renderSceneToImage(image, bvh, camera, eye, center);
//         //     updateTexture(textureID, image);
//         //     cameraMoved = false;  // reset flag
//         // }

//         glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//         glClear(GL_COLOR_BUFFER_BIT);

//         glUseProgram(shaderProgram);

//         glActiveTexture(GL_TEXTURE0);
//         glBindTexture(GL_TEXTURE_2D, textureID);

//         glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

//         glBindVertexArray(VAO);
//         glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//         glBindVertexArray(0);

//         glUseProgram(0);

//         glViewport(0, 0, WIDTH, HEIGHT);

//         glfwSwapBuffers(window);
//     }

//     glfwTerminate();
    return 0;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    float speed = 5.0f;

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        bool moved = false;

        if (key == GLFW_KEY_W)  { eye.z -= speed; moved = true; }
        if (key == GLFW_KEY_S)  { eye.z += speed; moved = true; }
        if (key == GLFW_KEY_A)  { eye.x -= speed; moved = true; }
        if (key == GLFW_KEY_D)  { eye.x += speed; moved = true; }
        if (key == GLFW_KEY_Q)  { eye.y += speed; moved = true; }
        if (key == GLFW_KEY_E)  { eye.y -= speed; moved = true; }

        if (key == GLFW_KEY_UP)    { center.z -= speed; moved = true; }
        if (key == GLFW_KEY_DOWN)  { center.z += speed; moved = true; }
        if (key == GLFW_KEY_LEFT)  { center.x -= speed; moved = true; }
        if (key == GLFW_KEY_RIGHT) { center.x += speed; moved = true; }

        if (moved) cameraMoved = true;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    float step = 1.f; 
    camera.focalPlaneDist += float(yoffset) * step;

    // std::cout << camera.focalPlaneDist << std::endl;

    camera.focalPlaneDist = std::max( camera.focalPlaneDist, camera.sensorDist );

    cameraMoved = true;
}