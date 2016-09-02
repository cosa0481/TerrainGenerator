/* Generator.h
 * Corin Sandford
 * 9/2016
 */

#include <pangolin/pangolin.h>

#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <ctime>

#ifdef USEGLEW
#include <GL/glew.h>
#endif
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>
#else

#include <GL/glut.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif

#define Cos(th) cos(3.1415926/180*(th))
#define Sin(th) sin(3.1415926/180*(th))

#define mapSize 128                         // Map size in grid spaces
#define Pi 3.14159265358979323846           // Pi constant
#define FLIMIT 25                           // Limit on number of any one kind of feature

int mode = 0;                               // 0 for grid, 1 for mesh texture
int axes = 0;                               // Display axes (1 -> show)
double asp = -1440.0f/900.0f;               // Aspect ratio
int dim = 50;                               // Dimensions
double Ox = 0, Oy = 0, Oz = 0;              // LookAt location
int X,Y;                                    // Last mouse location

int gain = 1;                               // Perlin noise generator gain
int octaves = 10;                           // Perlin noise generator octaves
float noiseDamp = 250.0;                    // Amount by which to dampen the perlin noise
int noiseEnable = 0;                        // Enable noise (1), disable noise (0)

int featureHeight = 10;
float gaussAmp = 5.0;                       // Amplitude of Gaussians
int muRange = 1;                            // 0 - muRange is the range for Gaussian means
int sigmaRange = 5;                         // 0 - sigmaRange is the range for Gaussian stdevs
int gmult = 5;                              // gmult * mapSize = number of Gaussians to make terrain
int quadSize = 16;                          // Distance between heig22ht map vertices
float zmax = 0;//+1e8;                      // Highest height map vertex
float zmin = 0;//-1e8;                      // Lowest height map vertex
float zmag = 1.0;                           // Height map value magnification
float z[mapSize][mapSize];                  // Height map
float zo[mapSize][mapSize];                 // Copy of height map with Gaussians only (no features)
float norms[mapSize][mapSize][3];           // Normal map
float texmap[mapSize][mapSize];             // Texture map (1 = green, 2 = dirt, 3 = metal)

int centerX = mapSize/2;                    // Center of the terrain map in the x-direction
int centerY = mapSize/2;                    // Center of the terrain map in the y-direction

// Lighting
int light = 1;                              // Light switch
float white[] = {1,1,1,1};                  // White vector
float black[] = {0,0,0,1};                  // Black vector
float yellow[] = {1.0,1.0,0.0,1.0};         // Yellow vector
int dist = 1000;                            // Light distance
int inc = 1;                                // Ball increment
int emission = 0;                           // Emission intensity (%)
int ambient = 30;                           // Ambient intensity (%)
int diffuse = 100;                          // Diffuse intensity (%)
int specular = 1;                           // Specular intensity (%)
int shininess = 7;                          // Shininess (power of 2)
float shinyvec[1] = {128};                  // Shininess (value)
int xlight = 90;                            // Light azimuthal angle
float ylight = 90;                          // Light elevation angle

// Mocha Gui Exporting
int scale = 100;    // Scale for reducing mesh for Mocha Gui
