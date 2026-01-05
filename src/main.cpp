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

const GLuint WIDTH = 400, HEIGHT = 400;

struct Pixel {
    glm::vec3 color=glm::vec3( 0. );
    glm::vec2 coord;

    Pixel( glm::vec2 coord ) : coord(coord) {}
    Pixel() : color( glm::vec3( 0. ) ) {}
};

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

    float cosTheta = (1.0f - u1) + u1 * cosThetaMax;

    float sinTheta = sqrt(1 - cosTheta * cosTheta);
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

// float shadow( glm::vec3 &ro, glm::vec3 &rd, float maxDist, BVH* bvh)
// {
//     Hit *hit = bvh->traverse( ro, rd, nullptr );
    
//     if( !hit ) return 1.f;
//     return glm::clamp( hit->t, 0.f, 1.f );
// }

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

enum LightType { SUN, POINT, AREA };

struct Light {
    public:
    glm::vec3 lightPos, Li;
    LightType type;

    Light( glm::vec3 lightPos, glm::vec3 Li, LightType type ) : 
    lightPos( lightPos ), 
    Li( Li ), 
    type( type ) {}

    float pdf() {
        switch( type ) {
            
            case LightType::SUN: {
                float cosThetaMax = .099;
                float pdf = 1.f/(2.f * PI * (1. - cosThetaMax));
                return pdf;
            };

            case LightType::POINT: {
                return 1.;
            }
            
            default: 
                return 0.;
        }
    }

    glm::vec3 sample( glm::vec3 pos, glm::vec3 nor, float seed ) {
        switch( type ) {

            // case LightType::AREA : {
            //     float u = hash( seed );
            //     float v = hash( seed );

            //     float r = sqrt( u );


            // }

            case LightType::SUN: {
                float cosThetaMax = .099;
                glm::vec3 sunDir = cosineDirection( seed, cosThetaMax, -lightPos);

                // Hit* sunHit = bvh->traverse(pos + nor * 1e-6f, sunDir, nullptr);

                // glm::vec3 Le = glm::vec3( 0.f );
                // if ( !sunHit ) {
                //     // Le = glm::vec3( 0. );
                //     sunDir = glm::vec3( 0. );
                // } else {
                //     float NdotL = std::max(glm::dot( nor, sunDir ), 0.f);
                //     // float pdf = 1.f/(2.f * PI * (1. - cosThetaMax));
                //     // glm::vec3 Li = lightCol * .7f;
                //     Le = bsdf * Li * NdotL / pdf;
                // }
                
                // delete sunHit;

                return sunDir;
            }

            case LightType::POINT: {

                glm::vec3 liray = glm::normalize( lightPos - pos );

                // Hit* lightHit = bvh->traverse(pos + nor * 1e-6f, liray, &tmax);

                // glm::vec3 Le( 0.f );
                // if ( !lightHit ) {
                //     Le = glm::vec3( 0. );
                // } else {
                //     float NdotL = std::max(glm::dot( nor, liray ), 0.f);
                //     // float pdf = 1.f; // cause point light lol
                //     // glm::vec3 Li = Li * .7f;
                //     Le = bsdf * Li * NdotL / pdf;
                // }

                // delete lightHit;

                return liray;
            }

            default:
                return glm::vec3( 0.f );
        }
    }

    glm::vec3 Le( glm::vec3 pos, glm::vec3 nor, glm::vec3 bsdf, glm::vec3 wi, float pdf, float seed, BVH *bvh, float tmax ) {
        switch( type ) {

            case LightType::SUN: {
                // float cosThetaMax = sqrt(3.)/2.f;
                glm::vec3 sunDir = wi;
                float NdotL = glm::dot( nor, sunDir );
                
                if( NdotL <= 0.f ) return glm::vec3( 0. );

                Hit sunHit;
                bool isHit = bvh->IntersectBVH(pos + nor * 1e-6f, sunDir, sunHit);

                glm::vec3 Le = glm::vec3( 0.f );
                if ( isHit ) {
                    Le = glm::vec3( 0. );
                } else {
                    // Le = glm::vec3( 0. );
                    // float pdf = 1.f/(2.f * PI * (1. - cosThetaMax));
                    // glm::vec3 Li = lightCol * .7f;
                    Le = bsdf * Li * NdotL / pdf;
                }
                
                // delete sunHit;

                return Le;
            }

            case LightType::POINT: {

                glm::vec3 d = lightPos - pos;
                glm::vec3 liray = wi;
                float r2 = glm::dot( d, d );

                Hit lightHit;
                bool isHit = bvh->IntersectBVH(pos + nor * 1e-6f, liray, lightHit, tmax);

                glm::vec3 Le( 0.f );
                if ( !isHit ) {
                    Le = glm::vec3( 0. );
                } else {
                    float NdotL = std::max(glm::dot( nor, liray ), 0.f);
                    // float pdf = 1.f; // cause point light lol
                    // glm::vec3 Li = Li * .7f;
                    float atten = 1.f / r2;
                    Le = bsdf * Li * NdotL * atten  / pdf;
                }

                // delete lightHit;

                return Le;
            }

            default:
                return glm::vec3( 0.f );
        }
    }
};

enum BSDFType { LAMBERT };

struct BSDF {
    glm::vec3 color;
    BSDFType type;

    BSDF( glm::vec3 color, BSDFType type ) : 
    color( color ), type( type ) {}

    glm::vec3 sample() {
        return color / float(PI);
    }

    float pdf( glm::vec3 wi ) {
        return 1. / (2. * PI);
    }
};

inline float PowerHeuristic( int nf, float fPdf, int ng, float gPdf ) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return ( f * f ) / ( f * f + g * g );
}

glm::vec3 rendererCalculateColor( glm::vec3& ro, glm::vec3& rd, BVH *bvh, float sa ) {
    
    glm::vec3 lightCol = glm::vec3(1.000, .8, 1.);
    // glm::vec3 lightCol = glm::vec3(1.000, .8, .6);
    // glm::vec3 lightPos = glm::vec3(280., 500., 250.);
    glm::vec3 lightPos = glm::vec3(-680., 1000., 250.);
    // glm::vec3 sunPos = glm::vec3(0.f, 0.f, 1.);
    
    float cosThetaMax = sqrt(3.)/2.f;
    // glm::vec3 sunCol = glm::vec3(1, 0.396, 0.659);
    glm::vec3 sunCol = glm::vec3(1, 0.878, 0.929);
    glm::vec3 sunPos = glm::normalize(glm::vec3( -1., 1., -1 ));
    
    glm::vec3 beta = glm::vec3( 1., 1., 1. );
    glm::vec3 L = glm::vec3( 0. );

    const int bounces = 7;
    
    static std::vector<Light> lights = {
        Light( lightPos, lightCol * 25000.0f, LightType::POINT ),
        // Light( lightPos, glm::vec3( 0.98, 0.549, 0.129 ) * 7.8f, LightType::POINT ),
        // Light( glm::vec3( 280., 500., 50. ), glm::vec3( 0, 0.851, 0.416 ) * 3.8f, LightType::POINT ),
        Light( sunPos, sunCol * 1.0f, LightType::SUN ),
        // Light( glm::vec3(300., 500., 250.), sunCol * 210.1f, LightType::POINT ),
    };

    float nLights = float(lights.size());
    std::uniform_int_distribution<> distrib(0, nLights - 1);

    for( int i = 0; i < bounces; i ++ ) {

        Hit hit;
        bool isHit = bvh->IntersectBVH(ro, rd, hit);

        if (!isHit) {
            // delete hit;
            // L += beta * glm::vec3(0.2, 0.2, 0.3);
            break;
        }

        glm::vec3& pos = hit.hitPoint;
        glm::vec3 nor = glm::normalize(hit.triangle->n);

        glm::vec3 point = lightPos;
        glm::vec3 lDir = point - pos;
        glm::vec3 liray = glm::normalize( lDir );
    
        int matId = hit.triangle->materialID;
        tinyobj::material_t& mat = Materials[matId];

        glm::vec3 color = glm::vec3(mat.diffuse[0], 
                                    mat.diffuse[1], 
                                    mat.diffuse[2]);
        glm::vec3 bsdf = color / float(PI);

        int lightInd = distrib( rng );

        // for( Light& light : lights ) {
        // }
        Light& light = lights[ lightInd ];
                
        // MIS from light
        glm::vec3 wi = light.sample( pos, nor, hash( sa ) );

        float cosThetaShadow = std::max(glm::dot( nor, wi ), 0.f);

        float gPdf = cosThetaShadow / (  PI);
        float fPdf = light.pdf();
        
        glm::vec3 Le = light.Le( pos, nor, bsdf, wi, fPdf, hash(sa), bvh, glm::dot( lDir, lDir ) + 1e-6f );

        if( light.type == POINT ) {
            L += beta * nLights * Le;
        } else {
            float weight = PowerHeuristic(1, fPdf, 1, gPdf);
            Le *= weight;
            L += beta * nLights * Le;
        }


        // // MIS from object
        // wi = light.sample( pos, nor, hash( sa ) );

        // cosThetaShadow = std::max(glm::dot( nor, wi ), 0.f);

        // fPdf = cosThetaShadow / ( PI);
        // gPdf = light.pdf();
        
        // Le = light.Le( pos, nor, bsdf, wi, gPdf, hash(sa), bvh, glm::dot(lDir, lDir) + 1e-6 );

        // if( light.type == POINT ) {
        //     L += beta * nLights * Le;
        // } else {
        //     float weight = PowerHeuristic(1, fPdf, 1, gPdf);
        //     Le *= weight;
        //     L += beta * nLights * Le;
        // }

        // Hit* lightHit = bvh->traverse(pos + nor * 1e-6f, liray, &hit->t);

        // if ( !lightHit ) {
        //     L += glm::vec3( 0. );
        // } else {
        //     float NdotL = std::max(glm::dot( nor, liray ), 0.f);
        //     float pdf = 1.f; // cause point light lol
        //     glm::vec3 Li = lightCol * .7f;
        //     L += beta * bsdf * Li * NdotL / pdf;
        // }
        
        // delete lightHit;

        // glm::vec3 sunDir = cosineDirection( hash( sa ), cosThetaMax, sunPos );
        // // sunDir

        // Hit* sunHit = bvh->traverse(pos + nor * 1e-6f, sunDir, nullptr);

        // if ( !sunHit ) {
        //     L += glm::vec3( 0. );
        // } else {
        //     float NdotL = std::max(glm::dot( nor, sunDir ), 0.f);
        //     float pdf = 1.f/(2.f * PI * (1. - cosThetaMax));
        //     glm::vec3 Li = sunCol * .7f;
        //     L += beta * bsdf * Li * NdotL / pdf;
        // }
        
        // delete sunHit;

        glm::vec3 bounceDir = RandomUnitVectorInHemisphereOf( nor, hash(sa) );
        float cosTheta = std::max(glm::dot(nor, bounceDir), 0.f);
        // float pdf = 1. / ( 2. * PI );  
        float pdf = cosTheta / ( PI );  

        beta *= bsdf * cosTheta / pdf;

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
        ro = pos + nor * 1e-6f;

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

        for (int i = 0; i < SAMPLES; i++)
        {
            Lens worldDir = camera.thinLensRay( st, float(i));
            // Hit* hit = bvh->traverse(worldDir.point, worldDir.dir, nullptr);

            pixel.color += rendererCalculateColor( worldDir.point, worldDir.dir, bvh, hash( i ) );
        }

        pixel.color /= float(SAMPLES);
        int idx = y * WIDTH + x;
        image[idx] = pixel;
    }
}

void renderSceneToImage(std::vector<Pixel>& image, BVH* bvh, Camera& camera, glm::vec3& eye, glm::vec3& center)
{
    int GRID_SIZE = WIDTH * HEIGHT;
    int SAMPLES = 1;
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

// bunny
// glm::vec3 center(-800.0f, 570.4f, 1000.0f);
// glm::vec3 eye(-100.0f, 560.4f, -100.0f);
// Camera camera( 1.5f, 840.9f, 1.15f, eye, center );

// Room
glm::vec3 center(-100.0f, 70.4f, -100.0f);
glm::vec3 eye(-100.0f, 60.4f, -500.0f);
Camera camera( 1.5f, 840.9f, 1.15f, eye, center );

bool cameraMoved = true; 

int main()
{
    LoadObj();

    BVH::initPool( triangles.size() );

    BVH* bvh = new BVH(triangles.begin(), triangles.end());
    
    std::cout << "built bvh" << std::endl;
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