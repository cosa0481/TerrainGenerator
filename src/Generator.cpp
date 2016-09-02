/* Generator.cpp
 * Corin Sandford
 * 9/2016
 * Generates and displays a height map that is editable using an attached gui
 */

#include <TerrainGenerator/Generator.h>

// Draw a Sun
static void Sun(double x,double y,double z,double r)
{
   int az,el;
   float Emission[]  = {0.0,0.0,float(0.01*emission),1.0};
   //  Save transformation
   glPushMatrix();
   //  Offset, scale and rotate
   glTranslated(x,y,z);
   glScaled(r,r,r);
   // Yellow ball
   glMaterialfv(GL_FRONT,GL_SHININESS,shinyvec);
   glMaterialfv(GL_FRONT,GL_SPECULAR,yellow);
   glMaterialfv(GL_FRONT,GL_EMISSION,Emission);
   // Set texture
   //glEnable(GL_TEXTURE_2D);
   //glBindTexture(GL_TEXTURE_2D, suntex);
   //  Bands of latitude
   for (el=-90;el<90;el+=inc) {
       glBegin(GL_QUAD_STRIP);
       for (az=0;az<=360;az+=2*inc) {
          double x = Sin(az)*Cos(el);
            double y = Cos(az)*Cos(el);
            double z =         Sin(el);
            //  For a sphere at the origin, the position
            //  and normal vectors are the same
            glNormal3d(x,y,z);
            //glTexCoord2d(az/360.0, el/180.0 + 0.5);
            glVertex3d(x,y,z);

            x = Sin(az)*Cos(el + 1);
            y = Cos(az)*Cos(el + 1);
            z =         Sin(el + 1);
            glNormal3d(x,y,z);
            //glTexCoord2d(az/360.0, (el + 1)/180.0 + 0.5);
            glVertex3d(x,y,z);
       }
       glEnd();
   }
   //glDisable(GL_TEXTURE_2D);
   //  Undo transofrmations
   glPopMatrix();
}

#define LEN 8192  //  Maximum length of text string
void Print(const char* format , ...)
{
   char buf[LEN];
   char* ch=buf;
   va_list args;

   //  Turn the parameters into a character string
   va_start(args,format);
   vsnprintf(buf,LEN,format,args);
   va_end(args);

   //  Display the characters one at a time at the current raster position
   while (*ch)
      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,*ch++);
}

float noise(int x, int y) {
   int n;

   n = x + y * 57;
   n = (n << 13) ^ n;

   return (1.0 - ( (n * ((n * n * 15731) + 789221) +  1376312589) & 0x7fffffff) / 1073741824.0);
}

float smooth_noise(int x, int y) {
   float corners;
   float center;
   float sides;

   corners = (noise(x - 1, y - 1) + noise(x + 1, y - 1) + noise(x - 1, x + 1) + noise(x + 1, y + 1)) / 16;
   sides = (noise(x - 1, y) + noise(x + 1, y) + noise(x, y - 1) + noise(x, y + 1)) / 8;
   center = noise(x, y) / 4;

   return (corners + sides + center);
}

float interpolate(float a, float b, float x) {
   float pi_mod;
   float f_unk;

   pi_mod = x * 3.1415927;
   f_unk = (1 - cos(pi_mod)) * 0.5;

   return (a * (1 - f_unk) + b * x);
}

float noise_handler(float x, float y) {
   int int_val[2];
   float frac_val[2];
   float value[4];
   float res[2];

   int_val[0] = (int)x;
   int_val[1] = (int)y;
   frac_val[0] = x - int_val[0];
   frac_val[1] = y - int_val[1];
   value[0] = smooth_noise(int_val[0], int_val[1]);
   value[1] = smooth_noise(int_val[0] + 1, int_val[1]);
   value[2] = smooth_noise(int_val[0], int_val[1] + 1);
   value[3] = smooth_noise(int_val[0] + 1, int_val[1] + 1);
   res[0] = interpolate(value[0], value[1], frac_val[0]);
   res[1] = interpolate(value[2], value[3], frac_val[0]);

   return (interpolate(res[0], res[1], frac_val[1]));
}

float perlin2(int x, int y, float g, int o) {
   int i;
   float total = 0.0f;
   float frequency = 1.0f/2.0;
   float amplitude = g;
   float lacunarity = 2.0;

   for (i = 0; i < o; ++i)
   {
     total += noise_handler((float)x * frequency, (float)y * frequency) * amplitude;
     frequency *= lacunarity;
     amplitude *= g;
   }

   return (total);
}

// Gaussian normal distribution
void gauss(float mu, float sigma, int x, int y, float p) {
   float A = gaussAmp;
   float zvalue;
   int i,j;

   for (i = 0; i < mapSize; i++) {
      for (j = 0; j < mapSize; j++) {
         // Equation for Gaussian Normal Distribution in 2-D
         zvalue = A * exp(-(((((x-i)-mu)*((x-i)-mu))/(2*sigma*sigma))+((((y-j)-mu)*((y-j)-mu))/(2*sigma*sigma))));
         z[i][j] += zvalue;// + noiseE*perlin2(i, j, gain, octaves)/noiseDamp;
         texmap[i][j] = 1;
         if (z[i][j] > zmax) zmax = z[i][j];
         if (z[i][j] < zmin) zmin = z[i][j];
      }
   }
   for (i = 0; i < mapSize; i++) {
       for (j = 0; j < mapSize; j++) {
           z[i][j] -= zmin;
           zo[i][j] = z[i][j];
       }
   }
}

// Generate height map by placing Gaussian distributions randomly around a grid
// Usually seeded with time(NULL)
void heightmap(unsigned int seed) {
   int k;
   int mu;
   int sigma = 0;
   srand(seed);

   for (k = 0; k <= gmult*mapSize; k++) {
      mu = rand() % muRange;
      sigma = 5 + rand() % sigmaRange;
      gauss(mu, sigma, rand() % (mapSize), rand() % (mapSize), 0);
   }
}

// Find normals for terrain height map z[][]
void findNormals() {

   int i,j;
   float nX, nY, nZ, n;
   float v1, v3, v4, v5, v7;
   float q1, q1x, q1y, q1z, q2, q2x, q2y, q2z, q3, q3x, q3y, q3z, q4, q4x, q4y, q4z;
   float v1x, v1y, v1z, v3x, v3y, v3z, v5x, v5y, v5z, v7x, v7y, v7z;

   /* Schematic for vertices used to
             * compute normal for vertex v4
             * v0------v1-------v2
             * |        ^        |
             * |   Q1   |   Q2   |
             * |        |        |
             * v3<-----v4------>v5
             * |        |        |
             * |   Q3   |   Q4   |
             * |        V        |
             * v6------v7-------v8
             */

   for (i = 0; i < mapSize; i++) {
      for (j = 0; j < mapSize; j++) {
         // Define vertices by height
         v1 = z[i][j+1];
         v3 = z[i-1][j];
         v4 = z[i][j];
         v5 = z[i+1][j];
         v7 = z[i][j-1];

         // Vector v4 -> v1
         v1x = 0.0;
         v1y = 1.0;
         v1z = v1 - v4;

         // Vector v4 -> v3
         v3x = -1.0;
         v3y = 0.0;
         v3z = v3 - v4;

         // Vector v4 -> v5
         v5x = 1.0;
         v5y = 0.0;
         v5z = v5 - v4;

         // Vector v4 -> v7
         v7x = 0.0;
         v7y = -1.0;
         v7z = v7 - v4;

         // Compute normal for Q1 with v41 x v43
         q1x = v1y*v3z - v1z*v3y;
         q1y = v1z*v3x - v1x*v3z;
         q1z = v1x*v3y - v1y*v3x;
         // Normalize Q1
         q1  = sqrt((q1x*q1x)+(q1y*q1y)+(q1z*q1z));
         q1x = q1x/q1;
         q1y = q1y/q1;
         q1z = q1z/q1;

         // Compute normal for Q2 with v45 x v41
         q2x = v5y*v1z - v5z*v1y;
         q2y = v5z*v1x - v5x*v1z;
         q2z = v5x*v1y - v5y*v1x;
         // Normalize Q2
         q2  = sqrt((q2x*q2x)+(q2y*q2y)+(q2z*q2z));
         q2x = q2x/q2;
         q2y = q2y/q2;
         q2z = q2z/q2;

         // Compute normal for Q3 with v43 x v47
         q3x = v3y*v7z - v3z*v7y;
         q3y = v3z*v7x - v3x*v7z;
         q3z = v3x*v7y - v3y*v7x;
         // Normalize Q3
         q3  = sqrt((q3x*q3x)+(q3y*q3y)+(q3z*q3z));
         q3x = q3x/q3;
         q3y = q3y/q3;
         q3z = q3z/q3;

         // Compute normal for Q4 with v47 x v45
         q4x = v7y*v5z - v7z*v5y;
         q4y = v7z*v5x - v7x*v5z;
         q4z = v7x*v5y - v7y*v5x;
         // Normalize Q4
         q4  = sqrt((q4x*q4x)+(q4y*q4y)+(q4z*q4z));
         q4x = q4x/q4;
         q4y = q4y/q4;
         q4z = q4z/q4;

         // Average the Q1, Q2, Q3, and Q4 to get the normal for v4
         nX = (q1x + q2x + q3x + q4x)/4;
         nY = (q1y + q2y + q3y + q4y)/4;
         nZ = (q1z + q2z + q3z + q4z)/4;

         // Normalize the normal for v4
         n = sqrt((nX*nX) + (nY*nY) + (nZ*nZ));
         nX = nX/n;
         nY = nY/n;
         nZ = nZ/n;

         // Save the normals to the corresponding height map position
         norms[i][j][0] = nX;
         norms[i][j][1] = nY;
         norms[i][j][2] = nZ;
      }
   }
}

/*************************************************************************************
 **                                    Features                                     **
 *************************************************************************************/

// Draw a hill using a gaussian with height h of mean mu, stdev sigma at point (x,y)
void hill(float h, float mu, float sigma, int x, int y, float p)
{
   float A = h;
   float zvalue;
   int i,j;

   for (i = 0; i < mapSize; i++) {
      for (j = 0; j < mapSize; j++) {
         // Equation for Gaussian Normal Distribution in 2-D
         zvalue = A * exp(-(((((x-i)-mu)*((x-i)-mu))/(2*sigma*sigma))+((((y-j)-mu)*((y-j)-mu))/(2*sigma*sigma))));
         if (zvalue > z[i][j]) {
            z[i][j] = zvalue;
            texmap[i][j] = 2;
         }
         if (z[i][j] < zmin) zmin = z[i][j];
         if (z[i][j] > zmax) zmax = z[i][j];
      }
   }
}

// Draw a valley using an upside down gaussian with height h of mean mu, stdev sigma at point (x,y)
void valley(float h, float mu, float sigma, int x, int y, float p, int b)
{
   float A = h;
   int base = b;
   float zvalue;
   int i,j;

   for (i = 0; i < mapSize; i++) {
      for (j = 0; j < mapSize; j++) {
         zvalue = base - (A * exp(-(((((x-i)-mu)*((x-i)-mu))/(2*sigma*sigma))+((((y-j)-mu)*((y-j)-mu))/(2*sigma*sigma)))));
         if (zvalue < z[i][j]) {
            z[i][j] = zvalue;
            texmap[i][j] = 2;
         }
         if (z[i][j] < zmin) zmin = z[i][j];
         if (z[i][j] > zmax) zmax = z[i][j];
      }
   }
}

// Draw a square plateau of height h, radius in each of 4 directions r and at point (x,y)
void plateau(int h, int lx, int ly, int x, int y)
{
   int i,j;
   int xlength = lx;
   int ylength = ly;
   int height = h;
   if (h <= 0) height = featureHeight;
   int xcenter = x % mapSize;
   int ycenter = y % mapSize;

   for (i = -xlength/2; i <= xlength/2; i++) {
      for (j = -ylength/2; j <= ylength/2; j++) {
         if (xcenter + i < mapSize && xcenter - i >= 0 && ycenter + j < mapSize && ycenter - j >= 0) {
            if (z[xcenter + i][ycenter + j] < height) {
               z[xcenter + i][ycenter + j] = height;
               texmap[xcenter + i][ycenter + j] = 2;
               texmap[xcenter + i - 1][ycenter + j] = 2;
               texmap[xcenter + i][ycenter + j - 1] = 2;
            }
         }
      }
   }
}

// Draw a plateau of width 3 of height h of length l in direction north (n) east(e) south(s) west(w)
// With center at point (x,y)
void wall(int h, int l, int x, int y, char dir)
{
   int i,j;
   int length = l;
   int height = h;
   if (h<=0) height = featureHeight;
   int xcenter = x % mapSize;
   int ycenter = y % mapSize;

   // North/South is the Y direction
   if (dir == 'n' || dir == 's') {
      for (j = ycenter - length/2; j < ycenter + length/2; j++) {
         if (z[xcenter][j] < height && j > 0 && j < mapSize) {
            z[xcenter][j] = height;
            texmap[xcenter][j] = 3;
         }
         if (z[xcenter+1][j] < height && xcenter+1 > 0 && xcenter+1 < mapSize && j > 0 && j < mapSize) {
            z[xcenter+1][j] = height;
            texmap[xcenter+1][j] = 3;
         }
         if (z[xcenter-1][j] < height && xcenter-1 > 0 && xcenter+1 < mapSize && j > 0 && j < mapSize) {
            z[xcenter-1][j] = height;
            texmap[xcenter-1][j] = 3;
         }
         if (xcenter+2 < mapSize && j > 0 && j < mapSize)
            texmap[xcenter+2][j] = 3;
         if (xcenter-2 > 0 && j > 0 && j < mapSize)
            texmap[xcenter-2][j] = 3;
         if (j-1 > 0)
            texmap[xcenter][j-1] = 3;
         if (xcenter-1 > 0 && j-1 > 0)
            texmap[xcenter-1][j-1] = 3;
         if (xcenter+1 < mapSize && j-1 > 0)
            texmap[xcenter+1][j-1] = 3;
         if (xcenter-2 > 0 && j-1 > 0)
            texmap[xcenter-2][j-1] = 3;
      }
   }
   // East/West is the X direction
   else if (dir == 'e' || dir == 'w') {
      for (i = xcenter - length/2; i < xcenter + length/2; i++) {
         if (z[i][ycenter] < height && i > 0 && i < mapSize) {
            z[i][ycenter] = height;
         }
            texmap[i][ycenter] = 3;
         if (z[i][ycenter+1] < height && ycenter+1 < mapSize && i > 0 && i < mapSize) {
            z[i][ycenter+1] = height;
            texmap[i][ycenter+1] = 3;
         }
         if (z[i][ycenter-1] < height && ycenter-1 > 0 && i > 0 && i < mapSize) {
            z[i][ycenter-1] = height;
            texmap[i][ycenter-1] = 3;
         }
         if (ycenter+2 < mapSize && i > 0 && i < mapSize)
            texmap[i][ycenter+2] = 3;
         if (ycenter-2 > 0 && i > 0 && i < mapSize)
            texmap[i][ycenter-2] = 3;
         if (i-1 > 0)
            texmap[i-1][ycenter] = 3;
         if (ycenter-1 > 0 && i-1 > 0)
            texmap[i-1][ycenter-1] = 3;
         if (ycenter+1 < mapSize && i-1 > 0)
            texmap[i-1][ycenter+1] = 3;
         if (ycenter-2 > 0 && i-1 > 0)
            texmap[i-1][ycenter-2] = 3;
      }
   }
}

// Draw an inclined plane of height h of length l and width w in direction dir
// Beginning at point (x,y)
void incline(int h, int l, int w, int x, int y, char dir)
{
   int i,j,k;
   int length = l;
   int width = w;
   if (width%2 != 0) width += 1;
   int height = h;
   if (h<=0) height = featureHeight;
   int xcenter = x % mapSize;
   int ycenter = y % mapSize;

   k = 0;
   // North is in the positive Y direction
   if (dir == 'n') {
      for (j = ycenter; j < ycenter + length; j++) {
         for (i = xcenter - width/2; i < xcenter + width/2; i++) {
            if (i < mapSize && j < mapSize && i > 0 && j > 0 && k > z[i][j]) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i-1][j] = 3;
            }
         }
         k += height/length;
      }
   }
   else if (dir == 's') {
      for (j = ycenter; j > ycenter - length; j--) {
         for (i = xcenter - width/2; i < xcenter + width/2; i++) {
            if (i < mapSize && j < mapSize && i > 0 && j > 0 && k > z[i][j]) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i-1][j] = 3;
            }
         }
         k += height/length;
      }
   }
   // East is in the positive X direction
   else if (dir == 'e') {
      for (i = xcenter; i < xcenter + length; i++) {
         for (j = ycenter - width/2; j < ycenter + width/2; j++) {
            if (i < mapSize && j < mapSize && i > 0 && j > 0 && k > z[i][j]) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i][j-1] = 3;
            }
         }
         k += height/length;
      }
   }
   else if (dir == 'w') {
      for (i = xcenter; i > xcenter - length; i--) {
         for (j = ycenter - width/2; j < ycenter + width/2; j++) {
            if (i < mapSize && j < mapSize && i > 0 && j > 0 && k > z[i][j]) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i][j-1] = 3;
            }
         }
         k += height/length;
      }
   }
}

// Draw a quarter pipe with height h, length l, curvature c, direction dir, with center at (x,y)
void qpipe(int h, int l, int x, int y, char dir)
{
   int i,j;
   float k;
   int length = l;
   int height = h;
   if (h<=0) height = featureHeight;
   int xcenter = x % mapSize;
   int ycenter = y % mapSize;

   k = 0;
   if (dir == 'w') {
      for (i = xcenter; i <= xcenter+height; i++) {
         for (j = ycenter-length/2; j <= ycenter+length/2; j++) {
            k = 10*((float)height - (float)height*sin(acos((float)(i-xcenter)/height)));
            if (i < mapSize && j < mapSize && k > z[i][j] && j > 0) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i][j-1] = 3;
            }
         }
      }
   }
   else if (dir == 'e') {
      for (i = xcenter; i >= xcenter-height; i--) {
         for (j = ycenter-length/2; j <= ycenter+length/2; j++) {
            k = 10*((float)height - (float)height*sin(acos((float)(i-xcenter)/height)));
            if (i > 0 && j < mapSize && k > z[i][j] && j > 0) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i][j-1] = 3;
            }
         }
      }
   }
   else if (dir == 's') {
      for (i = xcenter-length/2; i <= xcenter+length/2; i++) {
         for (j = ycenter; j <= ycenter+height; j++) {
            k = 10*((float)height - (float)height*sin(acos((float)(j-ycenter)/height)));
            if (i < mapSize && j < mapSize && k > z[i][j] && i > 0) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i-1][j] = 3;
            }
         }
      }
   }
   else if (dir == 'n') {
      for (i = xcenter-length/2; i <= xcenter+length/2; i++) {
         for (j = ycenter; j >= ycenter-height; j--) {
            k = 10*((float)height - (float)height*sin(acos((float)(j-ycenter)/height)));
            if (i < mapSize && j > 0 && k > z[i][j] && i > 0) {
               z[i][j] = k;
               texmap[i][j] = 3;
               texmap[i-1][j] = 3;
            }
         }
      }
   }
}

// Draw n smooth bumps of height h and length l
// The bumps begin with the midpoint of the length at (x,y) and go in dir direction
//TODO: implement whoops for other 3 directions//
void whoops(int h, int n, int l, int x, int y, char dir)
{

   int scale = 5;
   int i,j,k;
   float q;
   int length = l;
   int height = h;
   if (h<=0) height = featureHeight;
   int xcenter = x % mapSize;
   int ycenter = y % mapSize;

   if (dir == 'e') {
      for (k = 0; k < n; k++) {
         xcenter -= height;
         for (i = xcenter; i <= xcenter+height; i++) {
            for (j = ycenter-length/2; j <= ycenter+length/2; j++) {
               q = scale*((float)height - (float)height*sin(acos((float)(i-xcenter)/height)));
               if (i < mapSize && j < mapSize && q > z[i][j]) {
                  z[i][j] = q;
                  texmap[i][j] = 2;
                  texmap[i][j-1] = 2;
               }
            }
         }
         xcenter += 1.5*height;

         for (i = xcenter-height; i <= xcenter+height; i++) {
            for (j = ycenter-length/2; j <= ycenter+length/2; j++) {
               q = scale*(3*(float)height/4 + (float)height*sin(acos(2*(float)(i-xcenter)/height)));
               if (i < mapSize && j < mapSize && q > z[i][j]) {
                  z[i][j] = q;
                  texmap[i][j] = 2;
                  texmap[i][j-1] = 2;
               }
            }
         }

         xcenter += 1.5*height;
         for (i = xcenter; i >= xcenter-height; i--) {
            for (j = ycenter-length/2; j <= ycenter+length/2; j++) {
               q = scale*((float)height - (float)height*sin(acos((float)(i-xcenter)/height)));
               if (i < mapSize && j < mapSize && q > z[i][j]) {
                  z[i][j] = q;
                  texmap[i][j] = 2;
                  texmap[i][j-1] = 2;
               }
            }
         }

      }
   }
}

int main( int /*argc*/, char* argv[] )
{  
    // Load configuration data
    pangolin::ParseVarsFile("app.cfg");

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Terrain Generator",1500,1000);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrixOrthographic(-2880,2800,-1800,1800,-3500,3500),
    pangolin::ModelViewLookAt(1,1,0.5, 0,0,0, pangolin::AxisZ)
    );

    const int UI_WIDTH = 180;

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), pangolin::Attach::ReversePix(UI_WIDTH), asp)
    .SetHandler(new pangolin::Handler3D(s_cam));

    // Add named Panel and bind to variables beginning 'Lui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel( "Lui" ).SetBounds( 0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH) );

    // Add second column to gui on right side
    pangolin::CreatePanel( "Rui" ).SetBounds( 0.0, 1.0, pangolin::Attach::ReversePix(UI_WIDTH), 1.0 );

    // Bind variables to buttons, check boxes, and sliders
    pangolin::Var<bool> reset_scene("Lui.Reset View",false,false);
    pangolin::Var<bool> axes_checkbox("Lui.Display Axes",false,true);
    pangolin::Var<bool> mode_checkbox("Lui.Grid/Mesh",false,true);
    pangolin::Var<bool> output_to_mesh("Lui.Ouput as Mesh",false,false);
    pangolin::Var<std::string> mesh_filename("Lui.Filename", "meshfilename.ply");

    // Hills
    pangolin::Var<bool> drawHill("Lui.Place Hill",false,false);
    pangolin::Var<float> hHeight("Lui.Hill Height",500,0,1000);
    pangolin::Var<float> hRadius("Lui.Hill Sigma",15,0,50);
    pangolin::Var<int> hX("Lui.Hill X Position",mapSize/2,0,mapSize);
    pangolin::Var<int> hY("Lui.Hill Y Position",mapSize/2,0,mapSize);
    pangolin::Var<bool> removeHill("Lui.Remove Hill",false,false);
    pangolin::Var<int> hillNo("Lui.Hill No.",0,0,FLIMIT);
    pangolin::Var<bool> showHills("Lui.Show Hills",false,false);

    // Valleys
    pangolin::Var<bool> drawValley("Lui.Place Valley",false,false);
    pangolin::Var<int> vBase("Lui.Top of Valley",500,0,1000);
    pangolin::Var<float> vHeight("Lui.Valley Height",500,0,1000);
    pangolin::Var<float> vRadius("Lui.Valley Sigma",15,0,50);
    pangolin::Var<int> vX("Lui.Valley X Position",mapSize/2,0,mapSize);
    pangolin::Var<int> vY("Lui.Valley Y Position",mapSize/2,0,mapSize);
    pangolin::Var<bool> removeValley("Lui.Remove Valley",false,false);
    pangolin::Var<int> valleyNo("Lui.Valley No.",0,0,FLIMIT);
    pangolin::Var<bool> showValleys("Lui.Show Valleys",false,false);

    // Ramps
    pangolin::Var<bool> drawRamp("Lui.Place Ramp",false,false);
    pangolin::Var<int> rHeight("Lui.Ramp Height",1000,0,2000);
    pangolin::Var<int> rLength("Lui.Ramp Length",0,0,mapSize);
    pangolin::Var<int> rWidth("Lui.Ramp Width",0,0,mapSize);
    pangolin::Var<int> rX("Lui.Ramp X Position",mapSize/2,0,mapSize);
    pangolin::Var<int> rY("Lui.Ramp Y Position",mapSize/2,0,mapSize);
    pangolin::Var<int> rDirection("Lui.Ramp Direction",0,0,3);
    pangolin::Var<bool> removeRamp("Lui.Remove Ramp",false,false);
    pangolin::Var<int> rampNo("Lui.Ramp No.",0,0,FLIMIT);
    pangolin::Var<bool> showRamps("Lui.Show Ramps",false,false);

    // Walls
    pangolin::Var<bool> drawWall("Rui.Place Wall",false,false);
    pangolin::Var<int> wHeight("Rui.Wall Height",500,0,1000);
    pangolin::Var<int> wLength("Rui.Wall Length",0,0,mapSize);
    pangolin::Var<int> wX("Rui.Wall X Position",mapSize/2,0,mapSize-2);
    pangolin::Var<int> wY("Rui.Wall Y Position",mapSize/2,0,mapSize-2);
    pangolin::Var<int> wDirection("Rui.Wall Direction",0,0,3);
    pangolin::Var<bool> removeWall("Rui.Remove Wall",false,false);
    pangolin::Var<int> wallNo("Rui.Wall No.",0,0,FLIMIT);
    pangolin::Var<bool> showWalls("Rui.Show Walls",false,false);

    // Plateaus
    pangolin::Var<bool> drawPlateau("Rui.Place Plateau",false,false);
    pangolin::Var<int> plHeight("Rui.Plateau Height",500,0,1000);
    pangolin::Var<int> plxLength("Rui.Plateau X Length",0,0,mapSize);
    pangolin::Var<int> plyLength("Rui.Plateau Y Length",0,0,mapSize);
    pangolin::Var<int> plX("Rui.Plateau X Position",mapSize/2,0,mapSize-2);
    pangolin::Var<int> plY("Rui.Plateau Y Position",mapSize/2,0,mapSize-2);
    pangolin::Var<bool> removePlateau("Rui.Remove Plateau",false,false);
    pangolin::Var<int> plateauNo("Rui.Plateau No.",0,0,FLIMIT);
    pangolin::Var<bool> showPlateaus("Rui.Show Plateaus",false,false);

    // Quarter Pipes
    pangolin::Var<bool> drawPipe("Rui.Place Pipe",false,false);
    pangolin::Var<int> pHeight("Rui.Pipe Height",30,0,100);
    pangolin::Var<int> pLength("Rui.Pipe Length",0,0,mapSize);
    pangolin::Var<int> pX("Rui.Pipe X Position",mapSize/2,0,mapSize);
    pangolin::Var<int> pY("Rui.Pipe Y Position",mapSize/2,0,mapSize);
    pangolin::Var<int> pDirection("Rui.Pipe Direction",0,0,3);
    pangolin::Var<bool> removePipe("Rui.Remove Pipe",false,false);
    pangolin::Var<int> pipeNo("Rui.Pipe No.",0,0,FLIMIT);
    pangolin::Var<bool> showPipes("Rui.Show Pipes",false,false);

    // Whoops
    pangolin::Var<bool> drawWhoop("Rui.Place Whoop",false,false);
    pangolin::Var<int> whHeight("Rui.Whoop Height",20,0,50);
    pangolin::Var<int> whN("Rui.Number of Whoops",0,0,20);
    pangolin::Var<int> whLength("Rui.Width of Whoops",25,0,mapSize);
    pangolin::Var<int> whX("Rui.Whoop X Position",mapSize/2,0,mapSize);
    pangolin::Var<int> whY("Rui.Whoop Y Position",mapSize/2,0,mapSize);
    pangolin::Var<int> whDirection("Rui.Whoop Direction",0,0,3);
    pangolin::Var<bool> removeWhoop("Rui.Remove Whoop",false,false);
    pangolin::Var<int> whoopNo("Rui.Whoop No.",0,0,FLIMIT);
    pangolin::Var<bool> showWhoops("Rui.Show Whoops",false,false);

    // Register key presses and change variables accordingly
    //pangolin::RegisterKeyPressCallback('+', [dim]() mutable {dim = dim + 10;});

    // Generate height map
    heightmap(time(NULL));

    // Arrays for holding features
    char dir = 'n';
    float hillList[FLIMIT][4] = {0};
    int h = 0;
    float valleyList[FLIMIT][5] = {0};
    int v = 0;
    int rampList[FLIMIT][6] = {0};
    int r = 0;
    int plateauList[FLIMIT][5] = {0};
    int pl = 0;
    int wallList[FLIMIT][5] = {0};
    int w = 0;
    int pipeList[FLIMIT][5] = {0};
    int p = 0;
    int whoopList[FLIMIT][6] = {0};
    int wh = 0;

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() )
    {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glShadeModel(GL_SMOOTH);

        // Handle button pushes
        if ( pangolin::Pushed(reset_scene) )
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(1,1,0.5, 0,0,0, pangolin::AxisZ));
        if ( axes_checkbox ) axes = 1;
        else axes = 0;
        if ( mode_checkbox ) mode = 1;
        else mode = 0;

        // Activate efficiently by object
        d_cam.Activate(s_cam);

        // Render the height map as a grid
        int i,j,a,b,c;
        float x,y;
        double z0 = (zmin + zmax) / 2;

        // Reset heightmap
        for (i = 0; i < mapSize; i++)
            for (j = 0; j < mapSize; j++)
                z[i][j] = zo[i][j];
        // Lighting
        float Ambient[] = {float(0.01*ambient), float(0.01*ambient), float(0.01*ambient), 1.0};
        float Diffuse[] = {float(0.01*diffuse), float(0.01*diffuse), float(0.01*diffuse), 1.0};
        float Specular[] = {float(0.01*specular), float(0.01*specular), float(0.01*specular), 1.0};
        // Light position and color
        float Position[] = {float(dist*Sin(xlight)), float(dist*Cos(xlight)), float(dist*Sin(ylight)), 1.0};
        // Draw light position as ball (still no lighting)
        glColor3f(1,1,1);
        Sun(Position[0], Position[1], Position[2], 10);
        // OpenGL should normalize normal vectors
        glEnable(GL_NORMALIZE);
        // Enable lighting
        glEnable(GL_LIGHTING);
        // Location of viewer for specular calculations
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
        // glColorMaterial sets ambient and diffuse color materials
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glEnable(GL_COLOR_MATERIAL);
        // Enable light 0
        glEnable(GL_LIGHT0);
        // Set ambient, diffuse, specular components and position of light 0
        glLightfv(GL_LIGHT0, GL_AMBIENT, Ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, Diffuse);
        glLightfv(GL_LIGHT0, GL_SPECULAR, Specular);
        glLightfv(GL_LIGHT0, GL_POSITION, Position);

        // Draw features on heightmap

        /* Hills */
        // hill(float h, float mu, float sigma, int x, int y, float p)
        //------------------------------------------------------------
        // Display all hills and parameters in stdout
        if ( pangolin::Pushed(showHills) )
        {
            std::cout << "HILLS" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (hillList[a][0] != 0.0)
                {
                    std::cout << "Hill: " << a << std::endl;
                    std::cout << "   height: " << hillList[a][0] << "\t";
                    std::cout << "    sigma: " << hillList[a][1] << std::endl;
                    std::cout << "xposition: " << hillList[a][2] << "\t";
                    std::cout << "yposition: " << hillList[a][3] << "\n" << std::endl;
                }
            }
        }
        // Remove hill
        if ( pangolin::Pushed(removeHill) )
        {
            for (b = 1; b <= (h - hillNo); b++ )
            {
                hillList[hillNo+b-1][0] = hillList[hillNo+b][0];
                hillList[hillNo+b-1][1] = hillList[hillNo+b][1];
                hillList[hillNo+b-1][2] = hillList[hillNo+b][2];
                hillList[hillNo+b-1][3] = hillList[hillNo+b][3];
            }
            hillList[h][0] = 0.0;
            hillList[h][1] = 0.0;
            hillList[h][2] = 0.0;
            hillList[h][3] = 0.0;
            if (h > 0) h--;
        }
        // Save new hill for drawing
        if ( pangolin::Pushed(drawHill) )
        {
            if (h < FLIMIT)
            {
                hillList[h][0] = hHeight;
                hillList[h][1] = hRadius;
                hillList[h][2] = float(hX);
                hillList[h][3] = float(hY);
                h++;
            } else {
                std::cout << "Maximum number of hill features reached." << std::endl;
            }
        }
        // Draw hills
        for (c = 0; c < h; c++)
        {
            hill( hillList[c][0], 1.0, hillList[c][1], int(hillList[c][2]), int(hillList[c][3]), 0.0 );
        }
        /*hill(400, 1, 15, 30, mapSize-30, 0);
        hill(500, 1, 20, centerX-50, centerY-20, 0);
        hill(400, 1, 15, centerX+50, centerY-30, 0);*/

        /* Valleys */
        // valley(float h, float mu, float sigma, int x, int y, float p)
        //--------------------------------------------------------------
        // Display all valleys and parameters in stdout
        if ( pangolin::Pushed(showValleys) )
        {
            std::cout << "VALLEYS" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (valleyList[a][0] != 0.0)
                {
                    std::cout << "Valley: " << a << std::endl;
                    std::cout << "   height: " << valleyList[a][0] << "\t";
                    std::cout << "  bheight: " << valleyList[a][4] << "\t";
                    std::cout << "    sigma: " << valleyList[a][1] << std::endl;
                    std::cout << "xposition: " << valleyList[a][2] << "\t";
                    std::cout << "yposition: " << valleyList[a][3] << "\n" << std::endl;
                }
            }
        }
        // Remove valley
        if ( pangolin::Pushed(removeValley) )
        {
            for (b = 1; b <= (v - valleyNo); b++ )
            {
                valleyList[valleyNo+b-1][0] = valleyList[valleyNo+b][0];
                valleyList[valleyNo+b-1][1] = valleyList[valleyNo+b][1];
                valleyList[valleyNo+b-1][2] = valleyList[valleyNo+b][2];
                valleyList[valleyNo+b-1][3] = valleyList[valleyNo+b][3];
                valleyList[valleyNo+b-1][4] = valleyList[valleyNo+b][4];
            }
            valleyList[v][0] = 0.0;
            valleyList[v][1] = 0.0;
            valleyList[v][2] = 0.0;
            valleyList[v][3] = 0.0;
            valleyList[v][4] = 0.0;
            if (v > 0) v--;
        }
        // Save new valley for drawing
        if ( pangolin::Pushed(drawValley) )
        {
            if (v < FLIMIT)
            {
                valleyList[v][0] = vHeight;
                valleyList[v][1] = vRadius;
                valleyList[v][2] = float(vX);
                valleyList[v][3] = float(vY);
                valleyList[v][4] = vBase;
                v++;
            } else {
                std::cout << "Maximum number of valley features reached." << std::endl;
            }
        }
        // Draw valleys
        for (c = 0; c < v; c++)
        {
            valley( valleyList[c][0], 1.0, valleyList[c][1], int(valleyList[c][2]), int(valleyList[c][3]), 0.0, valleyList[c][4] );
        }
        //valley(1000, 1, 15, centerX-65, centerY-35, 0);

        /* Inclines */
        // incline(int h, int l, int w, int x, int y, char dir)
        //-----------------------------------------------------
        // Display all ramps and parameters in stdout
        if ( pangolin::Pushed(showRamps) )
        {
            std::cout << "RAMPS" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (rampList[a][0] != 0)
                {
                    dir = 'n';
                    if (rampList[a][5] == 1) dir = 'e';
                    else if (rampList[a][5] == 2) dir = 's';
                    else if (rampList[a][5] == 3) dir = 'w';
                    std::cout << "Ramp: " << a << std::endl;
                    std::cout << "   height: " << rampList[a][0] << "\t";
                    std::cout << "   length: " << rampList[a][1] << "\t";
                    std::cout << "    width: " << rampList[a][2] << std::endl;
                    std::cout << "xposition: " << rampList[a][3] << "\t";
                    std::cout << "yposition: " << rampList[a][4] << "\t";
                    std::cout << "direction: " << dir << "\n" << std::endl;
                }
            }
        }
        // Remove ramp
        if ( pangolin::Pushed(removeRamp) )
        {
            for (b = 1; b <= (r - rampNo); b++ )
            {
                rampList[rampNo+b-1][0] = rampList[rampNo+b][0];
                rampList[rampNo+b-1][1] = rampList[rampNo+b][1];
                rampList[rampNo+b-1][2] = rampList[rampNo+b][2];
                rampList[rampNo+b-1][3] = rampList[rampNo+b][3];
                rampList[rampNo+b-1][4] = rampList[rampNo+b][4];
                rampList[rampNo+b-1][5] = rampList[rampNo+b][5];
            }
            rampList[r][0] = 0;
            rampList[r][1] = 0;
            rampList[r][2] = 0;
            rampList[r][3] = 0;
            rampList[r][4] = 0;
            rampList[r][5] = 0;
            if (r > 0) r--;
        }
        // Save new ramp for drawing
        if ( pangolin::Pushed(drawRamp) )
        {
            if (r < FLIMIT)
            {
                rampList[r][0] = rHeight;
                rampList[r][1] = rLength;
                rampList[r][2] = rWidth;
                rampList[r][3] = rX;
                rampList[r][4] = rY;
                rampList[r][5] = rDirection;
                r++;
            } else {
                std::cout << "Maximum number of ramp features reached." << std::endl;
            }
        }
        // Draw ramps
        for (c = 0; c < r; c++)
        {
            dir = 'n';
            if (rampList[c][5] == 1) dir = 'e';
            else if (rampList[c][5] == 2) dir = 's';
            else if (rampList[c][5] == 3) dir = 'w';
            incline( rampList[c][0], rampList[c][1], rampList[c][2], rampList[c][3], rampList[c][4], dir );
        }
        /*incline(400, 55, 25, mapSize-30, 25, 'n');
        incline(400, 100, 25, mapSize-30, mapSize-50, 's');*/

        /* Walls */
        // wall(int h, int l, int x, int y, char dir)
        //-------------------------------------------
        // Display all walls and parameters in stdout
        if ( pangolin::Pushed(showWalls) )
        {
            std::cout << "WALLS" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (wallList[a][0] != 0)
                {
                    dir = 'n';
                    if (wallList[a][4] == 1) dir = 'e';
                    else if (wallList[a][4] == 2) dir = 's';
                    else if (wallList[a][4] == 3) dir = 'w';
                    std::cout << "Wall: " << a << std::endl;
                    std::cout << "   height: " << wallList[a][0] << "\t";
                    std::cout << "   length: " << wallList[a][1] <<std::endl;
                    std::cout << "xposition: " << wallList[a][3] << "\t";
                    std::cout << "yposition: " << wallList[a][4] << "\t";
                    std::cout << "direction: " << dir << "\n" << std::endl;
                }
            }
        }
        // Remove wall
        if ( pangolin::Pushed(removeWall) )
        {
            for (b = 1; b <= (w - wallNo); b++ )
            {
                wallList[wallNo+b-1][0] = wallList[wallNo+b][0];
                wallList[wallNo+b-1][1] = wallList[wallNo+b][1];
                wallList[wallNo+b-1][2] = wallList[wallNo+b][2];
                wallList[wallNo+b-1][3] = wallList[wallNo+b][3];
                wallList[wallNo+b-1][4] = wallList[wallNo+b][4];
            }
            wallList[w][0] = 0;
            wallList[w][1] = 0;
            wallList[w][2] = 0;
            wallList[w][3] = 0;
            wallList[w][4] = 0;
            if (w > 0) w--;
        }
        // Save new wall for drawing
        if ( pangolin::Pushed(drawWall) )
        {
            if (w < FLIMIT)
            {
                wallList[w][0] = wHeight;
                wallList[w][1] = wLength;
                wallList[w][2] = wX;
                wallList[w][3] = wY;
                wallList[w][4] = wDirection;
                w++;
            } else {
                std::cout << "Maximum number of wall features reached." << std::endl;
            }
        }
        // Draw walls
        for (c = 0; c < w; c++)
        {
            dir = 'n';
            if (wallList[c][4] == 1) dir = 'e';
            else if (wallList[c][4] == 2) dir = 's';
            else if (wallList[c][4] == 3) dir = 'w';
            wall( wallList[c][0], wallList[c][1], wallList[c][2], wallList[c][3], dir );
        }
        /*wall(500, 257, centerX, mapSize-2, 'e');
        wall(500, 257, centerX, 2, 'w');
        wall(500, 257, mapSize-2, centerY, 'n');
        wall(500, 257, 2, centerY, 's');
        wall(500, 50, 25, centerY-40, 'e');
        wall(500, 100, centerX+50, 50, 'n');
        wall(500, 70, centerX-12, centerY+16, 'n');*/

        /* Plateaus */
        // plateau(int h, int lx, int ly, int x, int y)
        //------------------------------------
        // Display all plateaus and parameters in stdout
        if ( pangolin::Pushed(showPlateaus) )
        {
            std::cout << "PLATEAUS" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (plateauList[a][0] != 0)
                {
                    std::cout << "Plateau: " << a << std::endl;
                    std::cout << "   height: " << plateauList[a][0] << "\t";
                    std::cout << "  xlength: " << plateauList[a][1] << "\t";
                    std::cout << "  ylength: " << plateauList[a][2] << std::endl;
                    std::cout << "xposition: " << plateauList[a][3] << "\t";
                    std::cout << "yposition: " << plateauList[a][4] << std::endl;
                }
            }
        }
        // Remove plateau
        if ( pangolin::Pushed(removePlateau) )
        {
            for (b = 1; b <= (pl - plateauNo); b++ )
            {
                plateauList[plateauNo+b-1][0] = plateauList[plateauNo+b][0];
                plateauList[plateauNo+b-1][1] = plateauList[plateauNo+b][1];
                plateauList[plateauNo+b-1][2] = plateauList[plateauNo+b][2];
                plateauList[plateauNo+b-1][3] = plateauList[plateauNo+b][3];
                plateauList[plateauNo+b-1][4] = plateauList[plateauNo+b][4];
            }
            plateauList[pl][0] = 0;
            plateauList[pl][1] = 0;
            plateauList[pl][2] = 0;
            plateauList[pl][3] = 0;
            plateauList[pl][4] = 0;
            if (pl > 0) pl--;
        }
        // Save new plateau for drawing
        if ( pangolin::Pushed(drawPlateau) )
        {
            if (pl < FLIMIT)
            {
                plateauList[pl][0] = plHeight;
                plateauList[pl][1] = plxLength;
                plateauList[pl][2] = plyLength;
                plateauList[pl][3] = plX;
                plateauList[pl][4] = plY;
                pl++;
            } else {
                std::cout << "Maximum number of plateau features reached." << std::endl;
            }
        }
        // Draw walls
        for (c = 0; c < pl; c++)
        {
            plateau( plateauList[c][0], plateauList[c][1], plateauList[c][2], plateauList[c][3], plateauList[c][4] );
        }
        //plateau(400, 13, mapSize-30, centerY-35);

        /* Quarter Pipes */
        // qpipe(int h, int l, int x, int y, char dir)
        //--------------------------------------------
        // Display all qpipes and parameters in stdout
        if ( pangolin::Pushed(showPipes) )
        {
            std::cout << "QUARTER PIPES" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (pipeList[a][0] != 0)
                {
                    dir = 'n';
                    if (pipeList[a][4] == 1) dir = 'e';
                    else if (pipeList[a][4] == 2) dir = 's';
                    else if (pipeList[a][4] == 3) dir = 'w';
                    std::cout << "Quarter Pipe: " << a << std::endl;
                    std::cout << "   height: " << pipeList[a][0] << "\t";
                    std::cout << "   length: " << pipeList[a][1] <<std::endl;
                    std::cout << "xposition: " << pipeList[a][3] << "\t";
                    std::cout << "yposition: " << pipeList[a][4] << "\t";
                    std::cout << "direction: " << dir << "\n" << std::endl;
                }
            }
        }
        // Remove qpipe
        if ( pangolin::Pushed(removePipe) )
        {
            for (b = 1; b <= (p - pipeNo); b++ )
            {
                pipeList[pipeNo+b-1][0] = pipeList[pipeNo+b][0];
                pipeList[pipeNo+b-1][1] = pipeList[pipeNo+b][1];
                pipeList[pipeNo+b-1][2] = pipeList[pipeNo+b][2];
                pipeList[pipeNo+b-1][3] = pipeList[pipeNo+b][3];
                pipeList[pipeNo+b-1][4] = pipeList[pipeNo+b][4];
            }
            pipeList[p][0] = 0;
            pipeList[p][1] = 0;
            pipeList[p][2] = 0;
            pipeList[p][3] = 0;
            pipeList[p][4] = 0;
            if (p > 0) p--;
        }
        // Save new qpipe for drawing
        if ( pangolin::Pushed(drawPipe) )
        {
            if (p < FLIMIT)
            {
                pipeList[p][0] = pHeight;
                pipeList[p][1] = pLength;
                pipeList[p][2] = pX;
                pipeList[p][3] = pY;
                pipeList[p][4] = pDirection;
                p++;
            } else {
                std::cout << "Maximum number of quarter pipe features reached." << std::endl;
            }
        }
        // Draw qpipes
        for (c = 0; c < p; c++)
        {
            dir = 'n';
            if (pipeList[c][4] == 1) dir = 'e';
            else if (pipeList[c][4] == 2) dir = 's';
            else if (pipeList[c][4] == 3) dir = 'w';
            qpipe( pipeList[c][0], pipeList[c][1], pipeList[c][2], pipeList[c][3], dir );
        }
        /*qpipe(65, 100, centerX+70, mapSize-75, 's');
        qpipe(40, 40, 30, mapSize-75, 'n');
        qpipe(40, 40, 30, centerY-27, 's');
        qpipe(50, 90, centerX+35, centerY, 's');*/


        /* Whoops */
        // whoops(int h, int n, int l, int x, int y, char dir)
        //----------------------------------------------------
        // Display all whoops and parameters in stdout
        if ( pangolin::Pushed(showWhoops) )
        {
            std::cout << "WHOOPS" << std::endl;
            std::cout << "------------------------------" << std::endl;
            for (a = 0; a < FLIMIT; a++) {
                if (whoopList[a][0] != 0)
                {
                    dir = 'n';
                    if (whoopList[a][5] == 1) dir = 'e';
                    else if (whoopList[a][5] == 2) dir = 's';
                    else if (whoopList[a][5] == 3) dir = 'w';
                    std::cout << "Whoop: " << a << std::endl;
                    std::cout << "   height: " << whoopList[a][0] << "\t";
                    std::cout << "numwhoops: " << whoopList[a][1] << "\t";
                    std::cout << "   length: " << whoopList[a][2] <<std::endl;
                    std::cout << "xposition: " << whoopList[a][3] << "\t";
                    std::cout << "yposition: " << whoopList[a][4] << "\t";
                    std::cout << "direction: " << dir << "\n" << std::endl;
                }
            }
        }
        // Remove whoop
        if ( pangolin::Pushed(removeWhoop) )
        {
            for (b = 1; b <= (wh - whoopNo); b++ )
            {
                whoopList[whoopNo+b-1][0] = whoopList[whoopNo+b][0];
                whoopList[whoopNo+b-1][1] = whoopList[whoopNo+b][1];
                whoopList[whoopNo+b-1][2] = whoopList[whoopNo+b][2];
                whoopList[whoopNo+b-1][3] = whoopList[whoopNo+b][3];
                whoopList[whoopNo+b-1][4] = whoopList[whoopNo+b][4];
                whoopList[whoopNo+b-1][5] = whoopList[whoopNo+b][5];
            }
            whoopList[wh][0] = 0;
            whoopList[wh][1] = 0;
            whoopList[wh][2] = 0;
            whoopList[wh][3] = 0;
            whoopList[wh][4] = 0;
            whoopList[wh][5] = 0;
            if (wh > 0) wh--;
        }
        // Save new whoop for drawing
        if ( pangolin::Pushed(drawWhoop) )
        {
            if (wh < FLIMIT)
            {
                whoopList[wh][0] = whHeight;
                whoopList[wh][1] = whN;
                whoopList[wh][2] = whLength;
                whoopList[wh][3] = whX;
                whoopList[wh][4] = whY;
                whoopList[wh][5] = whDirection;
                wh++;
            } else {
                std::cout << "Maximum number of whoop features reached." << std::endl;
            }
        }
        // Draw whoops
        for (c = 0; c < wh; c++)
        {
            dir = 'n';
            if (whoopList[c][5] == 1) dir = 'e';
            else if (whoopList[c][5] == 2) dir = 's';
            else if (whoopList[c][5] == 3) dir = 'w';
            whoops(whoopList[c][0], whoopList[c][1], whoopList[c][2], whoopList[c][3], whoopList[c][4], dir);
        }
        /*whoops(20, 5, 25, centerX-100, mapSize-50, 'e');
        whoops(20, 3, 50, 50, 35, 'e');*/

        // Find normals for heightmap
        findNormals();

        // Write ply file header
        std::ofstream out;
        bool output = false;
        if ( pangolin::Pushed(output_to_mesh) ) {
            output = true;
            out.open( mesh_filename, std::ios::out ); // Creates new file by default
            if ( !out.is_open() ) std::cout << "Failed to open output file " << mesh_filename << std::endl;
            out << "ply" << std::endl;
            out << "format ascii 1.0" << std::endl;
            out << "comment generated by TerrainGenerator" << std::endl;
            out << "element vertex " << (mapSize * mapSize) << std::endl; // n x n vertices -> n-1 x n-1 squares
            out << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
            out << "element face " << (int)( ((mapSize-1) * (mapSize-1)) * 2 ) << std::endl; // n-1 x n-1 * 2 triangles
            out << "property list uchar int vertex_indices" << std::endl;
            out << "end_header" << std::endl;
        }

        // Display terrain wire frame
        if ( !mode ) {
           glColor3f(1,1,0);
           for (i = 0; i < mapSize; i++) {
              for (j = 0; j < mapSize; j++) {
                 x = quadSize * (i-centerX);
                 y = quadSize * (j-centerY);
                 glBegin(GL_LINE_LOOP);
                 glVertex3d(x+0,y+0,zmag*(z[i+0][j+0]-z0));
                 glVertex3d(x+quadSize,y+0,zmag*(z[i+1][j+0]-z0));
                 glVertex3d(x+quadSize,y+quadSize,zmag*(z[i+1][j+1]-z0));
                 glVertex3d(x+0,y+quadSize,zmag*(z[i+0][j+1]-z0));
                 glEnd();

                 // Write out vertices
                 if ( output ) {
                     out << std::to_string( (float)x/scale ) << " " << std::to_string( (float)y/scale ) << " " << std::to_string( (float)( (z[i+0][j+0] - zmin)/scale ) ) << std::endl;
                 }
              }
           }
        }
        //  Apply texture to terrain wireframe
        else {
           glColor3f(1,1,1);
           float Emission[] = {0.0,0.0,float(0.01*emission),1.0};
           glMaterialfv(GL_FRONT, GL_SHININESS, shinyvec);
           glMaterialfv(GL_FRONT, GL_SPECULAR, white);
           glMaterialfv(GL_FRONT, GL_EMISSION, Emission);
           //glEnable(GL_TEXTURE_2D);

           glEnable(GL_DEPTH_TEST);
           glEnable(GL_CULL_FACE);

           // Draw Quads
           for (i = 0; i < mapSize; i++) {
              for (j = 0; j < mapSize; j++) {

                 //if (texmap[i][j] == 2) glBindTexture(GL_TEXTURE_2D, dirt);
                 //else if (texmap[i][j] == 3) glBindTexture(GL_TEXTURE_2D, metal);
                 //else glBindTexture(GL_TEXTURE_2D, green);
                 // glBindTexture(GL_TEXTURE_2D, texture);

                 x = quadSize * (i-centerX);
                 y = quadSize * (j-centerY);

                 glBegin(GL_QUADS);
                 glNormal3d(norms[i][j][0], norms[i][j][1], norms[i][j][2]);
                 glTexCoord2f((i+0),(j+0));
                 glVertex3d((x+0),(y+0),zmag*(z[i+0][j+0]-z0));

                 glNormal3d(norms[i+1][j][0], norms[i+1][j][1], norms[i+1][j][2]);
                 glTexCoord2f((i+quadSize),(j+0));
                 glVertex3d((x+quadSize),(y+0),zmag*(z[i+1][j+0]-z0));

                 glNormal3d(norms[i+1][j+1][0], norms[i+1][j+1][1], norms[i+1][j+1][2]);
                 glTexCoord2f((i+quadSize),(j+quadSize));
                 glVertex3d((x+quadSize),(y+quadSize),zmag*(z[i+1][j+1]-z0));

                 glNormal3d(norms[i][j+1][0], norms[i][j+1][1], norms[i][j+1][2]);
                 glTexCoord2f((i+0),(j+quadSize));
                 glVertex3d((x+0),(y+quadSize),zmag*(z[i+0][j+1]-z0));
                 glEnd();

                 // Write out vertices
                 if ( output ) {
                     out << std::to_string( (float)x/scale ) << " " << std::to_string( (float)y/scale ) << " " << std::to_string( (float)( (z[i+0][j+0] - zmin)/scale ) ) << std::endl;
                 }
              }
           }
           glDisable(GL_CULL_FACE);
           glDisable(GL_DEPTH_TEST);
           //glDisable(GL_TEXTURE_2D);
        }

        // Write out triangles
        std::string v1, v2, v3, v4;
        for ( i = 0; i < mapSize * mapSize - mapSize; i++ ) {
            v1 = std::to_string( i );
            v2 = std::to_string( i+1 );
            v3 = std::to_string( i+mapSize );
            v4 = std::to_string( i+mapSize+1 );
            if ( (i+1) % mapSize != 0 ) {
                if ( output ) {
                    out << "3 " << v1 << " " << v2 << " " << v3 << std::endl;
                    out << "3 " << v2 << " " << v3 << " " << v4 << std::endl;
                }
            }
        }

        if ( out.is_open() ) out.close();

        // Show axes if required
        const double len = 150;
        glColor3f(1,1,1);
        if (axes) {
            glPushMatrix();
            glBegin(GL_LINES);
            glVertex3d(Ox,Oy,Oz);
            glVertex3d(Ox+len,Oy,Oz);
            glVertex3d(Ox,Oy,Oz);
            glVertex3d(Ox,Oy+len,Oz);
            glVertex3d(Ox,Oy,Oz);
            glVertex3d(Ox,Oy,Oz+len);
            glEnd();
            //  Label axes
            glRasterPos3d(Ox+len,Oy,Oz);
            Print("X");
            glRasterPos3d(Ox,Oy+len,Oz);
            Print("Y");
            glRasterPos3d(Ox,Oy,Oz+len);
            Print("Z");
            glPopMatrix();
        }

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
    }
