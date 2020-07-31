/*
 * Implementation of the Local Laplacian Filter algorithm as described in https://people.csail.mit.edu/sparis/publi/2011/siggraph/
 */

/*

    Copyright (C) 2020 Ferrero Andrea

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.


 */

/*

    These files are distributed with PhfMerge - https://github.com/aferrero2707/PhfMerge

 */

#include <vips/vips.h>
#include <vector>
#include <iostream>
#include <string>


#undef __SSE2__


static float THRESHOLD = 0.5f;
static float ALPHA = 1.5f;
static float ALPHA0 = 1.f;
static float BETA = 0.5f;
static float BETA0 = 0.5f;
#define STRENGTH 1.0f
#define LOGL 1


// NaN-safe clamping (NaN compares false, and will thus result in H)
#define CLAMPS(A, L, H) ((A) > (L) ? ((A) < (H) ? (A) : (H)) : (L))


#ifdef _WIN32
void dt_free_align(void *mem);
#define dt_free_align_ptr dt_free_align
#else
#define dt_free_align(A) free(A)
#define dt_free_align_ptr free
#endif

void *dt_alloc_align(size_t alignment, size_t size)
{
  void *ptr = NULL;
#if defined(__FreeBSD_version) && __FreeBSD_version < 700013
  ptr = malloc(size);
#elif defined(_WIN32)
  ptr = _aligned_malloc(size, alignment);
#else
  if(posix_memalign(&ptr, alignment, size)) return NULL;
#endif
  if( ptr ) memset(ptr, 0, size);
  return ptr;
}

#ifdef _WIN32
void dt_free_align(void *mem)
{
  _aligned_free(mem);
}
#endif


struct LLRect
{
  int left, top, width, height, right, bottom;
  void init()
  {
    right = left + width - 1;
    bottom = top + height - 1;
  }
};

class Image
{
public:
  std::string name;
  LLRect roi, valid;
  int width, height, pwidth, pheight, padding;
  float* pixels;
  float* buf;
  bool freebuf;

  Image(): width(0), height(0), pwidth(0), pheight(0), padding(0), pixels(NULL), buf(NULL), freebuf(false) {}
  ~Image()
  {
    if(buf && freebuf) {
      //std::cout<<"Freeing pixels from "<<name<<std::endl;
      dt_free_align(buf);
    }
  }

  void set_name(std::string n) { name = n; }

  void alloc_buffer(LLRect& v, LLRect& r)
  {
    float* tbuf = (float*)dt_alloc_align(64, sizeof(float)*(v.width)*(v.height));
    freebuf = true;
    set_buffer(v, r, tbuf);
  }

  void set_buffer(LLRect& v, LLRect& r, float* b)
  {
    roi = r;
    roi.right = roi.left + roi.width - 1;
    roi.bottom = roi.top + roi.height - 1;
    valid = v;
    valid.right = valid.left + valid.width - 1;
    valid.bottom = valid.top + valid.height - 1;

    //printf("set_buffer: roi.width=%d\n", roi.width);

    int dw = roi.left - valid.left;
    int dh = roi.top - valid.top;

    buf = b;
    pixels = &(buf[dh*valid.width + dw]);
  }

  void zero()
  {
    memset(buf, 0, sizeof(float)*pwidth*pheight);
  }

  void intersect(int& left, int& top, int& right, int& bottom)
  {
    if(left < 0) left = 0;
    if(top  < 0) top  = 0;
    if(right >= width) right = width - 1;
    if(bottom >= height) bottom = height - 1;
  }

  void pintersect(int& left, int& top, int& right, int& bottom)
  {
    if(left < -padding) left = -padding;
    if(top  < -padding) top  = -padding;
    if(right >= (width+padding)) right = width + padding - 1;
    if(bottom >= (height+padding)) bottom = height + padding - 1;
  }

  float get_pixel(int x, int y)
  {
    int dx = x - valid.left;
    int dy = y - valid.top;

    if(dx < 0) dx = 0;
    else if(dx >= valid.width) dx = valid.width - 1;

    if(dy < 0) dy = 0;
    else if(dy >= valid.height) dy = valid.height - 1;
    //printf("setting pixel %d,%d to %f\n", x, y, val);

    return buf[dy*valid.width+dx];
  }

  float operator ()(int x, int y)
  {
    return get_pixel(x, y);
  }

  void set_pixel(int x, int y, float val)
  {
    int dx = x - valid.left;
    int dy = y - valid.top;

    if(dx < 0) return;
    if(dx >= valid.width) return;

    if(dy < 0) return;
    else if(dy >= valid.height) return;

    //printf("setting pixel %d,%d to %f\n", x, y, val);
    //printf("setting pixel %d,%d from %f to %f\n", x, y, pixels[y*width+x], val);
    //printf("setting pixel %d,%d   buf=%p  pixels=%p  pixel=%p  delta=%d\n", x, y, buf, pixels, &(pixels[y*pwidth+x]), pixels-buf);
    buf[dy*valid.width+dx] = val;
  }

  void add_to_pixel(int x, int y, float val)
  {
    int dx = x - valid.left;
    int dy = y - valid.top;

    if(dx < 0) return;
    if(dx >= valid.width) return;

    if(dy < 0) return;
    else if(dy >= valid.height) return;

    //printf("setting pixel %d,%d to %f+%f=%f\n", x, y, get_pixel(x,y), val, get_pixel(x,y)+val);
    buf[dy*valid.width+dx] += val;
  }

  void add_image(Image& img, float sign=1.0f)
  {
    for(int y = 0; y < height; y++) {
      for(int x = 0; x < width; x++) {
        pixels[y*width+x] += img.pixels[y*width+x] * sign;
      }
    }
  }

  bool crop(LLRect c, LLRect r, Image& out)
  {
    //printf("crop: x0=%d y0=%d w=%d h=%d\n", c.left, c.top, c.width, c.height);
    c.init();

    // match the crop area to the valid region of the input image
    if(c.left < valid.left) c.left = valid.left;
    if(c.top < valid.top) c.top = valid.top;
    if(c.right > valid.right) c.right = valid.right;
    if(c.bottom > valid.bottom) c.bottom = valid.bottom;
    c.width = c.right - c.left + 1;
    c.height = c.bottom - c.top + 1;

    out.alloc_buffer(c, r);

    for(int y = 0; y < c.height; y++) {
      float* line = buf + (y+c.top) * valid.width + c.left;
      float* oline = out.buf + y * c.width;
      memcpy( oline, line, sizeof(float) * c.width );
    }
    //printf("Image::crop()\n");
    //print(); out.print();
    //print(x0+w/2, y0+h/2); out.print(w/2, h/2);
    return true;
  }

  bool copy(Image& out)
  {
    out.alloc_buffer(valid, roi);
    memcpy( out.buf, buf, sizeof(float)*valid.width*valid.height );
    return true;
  }

  void print_full()
  {
    for(int y = 0; y < height; y++) {
      //printf("%d\n", y);
      for(int x = 0; x < width; x++) {
        //printf("  %d: %f\n", x, pixels[y*width+x]);
        printf("\t%f", pixels[y*width+x]);
      }
      printf("\n");
    }

    //int y = dl(height, 1), x = dl(width, 1);
    //printf("%d %d -> %f\n", y, x, pixels[y*width+x]);
  }

  void print()
  {
    for(int x = valid.left; x <= valid.right; x++) {
      printf("      %03d   ", x);
    }
    printf("\n");
    //for(int y = valid.top; y <= valid.bottom; y++) {
    for(int y = (valid.top+valid.height/2); y <= (valid.top+valid.height/2); y++) {
      //printf("%d\n", y);
      for(int x = valid.left; x <= valid.right; x++) {
        //printf("  %d: %f\n", x, pixels[y*width+x]);
        if(x >= roi.left && x <= roi.right)
          printf("   %+0.5f ", get_pixel(x, y));
        else
          printf("  (%+0.5f)", get_pixel(x, y));
      }
      printf("\n");
    }

    //int y = dl(height, 1), x = dl(width, 1);
    //printf("%d %d -> %f\n", y, x, pixels[y*width+x]);
  }

  void print(int x, int y)
  {
    printf("%d %d -> %f\n", y, x, get_pixel(x, y));
  }
};


// downsample width/height to given level
static inline int dl(int size, const int level)
{
  for(int l=0;l<level;l++)
    size = (size-1)/2+1;
  return size;
}

// downsample left/top to given level
static inline int dl2(int size, const int level)
{
  for(int l=0;l<level;l++)
    size = size/2;
  return size;
}



// upsamnple width/height to given level
static inline int ul(int size, const int level)
{
  for(int l=0;l<level;l++)
    size = size * 2;
  return size;
}



// needs a boundary of 1 or 2px around i,j or else it will crash.
// (translates to a 1px boundary around the corresponding pixel in the coarse buffer)
// more precisely, 1<=i<wd-1 for even wd and
//                 1<=i<wd-2 for odd wd (j likewise with ht)
static inline float ll_expand_gaussian(
    const float *const coarse,
    int i,
    int j,
    int wd,
    int ht,
    bool verbose=false)
{
  //if(i < 1) i = 1;
  //if(i > (wd-3)) i = wd - 3;
  //if(j < 1) j = 1;
  //if(j > (ht)) j = ht - 3;
  assert(i > 0);
  assert(i < wd-1);
  assert(j > 0);
  assert(j < ht-1);
  assert(j/2 + 1 < (ht-1)/2+1);
  assert(i/2 + 1 < (wd-1)/2+1);
  const int cw = (wd-1)/2+1;
  const int ind = (j/2)*cw+i/2;
  const int type = (i&1) + 2*(j&1);
  if(verbose) printf("ll_expand_gaussian(%d, %d): wd=%d  ht=%d  cw=%d  ind=%d  type=%d\n", i, j, wd, ht, cw, ind, type);
  // case 0:     case 1:     case 2:     case 3:
  //  x . x . x   x . x . x   x . x . x   x . x . x
  //  . . . . .   . . . . .   . .[.]. .   .[.]. . .
  //  x .[x]. x   x[.]x . x   x . x . x   x . x . x
  //  . . . . .   . . . . .   . . . . .   . . . . .
  //  x . x . x   x . x . x   x . x . x   x . x . x
  switch((i&1) + 2*(j&1))
  {
  case 0: // both are even, 3x3 stencil
    if(verbose) printf("    %f %f %f\n", coarse[ind-cw-1], coarse[ind-cw], coarse[ind-cw+1]);
    if(verbose) printf("    %f %f %f\n", coarse[ind-1], coarse[ind], coarse[ind+1]);
    if(verbose) printf("    %f %f %f\n", coarse[ind+cw-1], coarse[ind+cw], coarse[ind+cw+1]);
    return 4./256. * (
        6.0f*(coarse[ind-cw] + coarse[ind-1] + 6.0f*coarse[ind] + coarse[ind+1] + coarse[ind+cw])
        + coarse[ind-cw-1] + coarse[ind-cw+1] + coarse[ind+cw-1] + coarse[ind+cw+1]);
  case 1: // i is odd, 2x3 stencil
    return 4./256. * (
        24.0*(coarse[ind] + coarse[ind+1]) +
        4.0*(coarse[ind-cw] + coarse[ind-cw+1] + coarse[ind+cw] + coarse[ind+cw+1]));
  case 2: // j is odd, 3x2 stencil
    return 4./256. * (
        24.0*(coarse[ind] + coarse[ind+cw]) +
        4.0*(coarse[ind-1] + coarse[ind+1] + coarse[ind+cw-1] + coarse[ind+cw+1]));
  default: // case 3: // both are odd, 2x2 stencil
    return .25f * (coarse[ind] + coarse[ind+1] + coarse[ind+cw] + coarse[ind+cw+1]);
  }
}



// needs a boundary of 1 or 2px around i,j or else it will crash.
// (translates to a 1px boundary around the corresponding pixel in the coarse buffer)
// more precisely, 1<=i<wd-1 for even wd and
//                 1<=i<wd-2 for odd wd (j likewise with ht)
static inline float ll_expand_gaussian(
    Image& coarse,
    int i,
    int j,
    bool verbose=false)
{
  const int i2 = i/2;
  const int j2 = j/2;
  const int type = (i&1) + 2*(j&1);
  if(verbose) printf("ll_expand_gaussian(%d, %d): type=%d\n", i, j, type);
  // case 0:     case 1:     case 2:     case 3:
  //  x . x . x   x . x . x   x . x . x   x . x . x
  //  . . . . .   . . . . .   . .[.]. .   .[.]. . .
  //  x .[x]. x   x[.]x . x   x . x . x   x . x . x
  //  . . . . .   . . . . .   . . . . .   . . . . .
  //  x . x . x   x . x . x   x . x . x   x . x . x
  switch(type)
  {
  case 0: // both are even, 3x3 stencil
    //if(verbose) printf("    %f %f %f\n", coarse[ind-cw-1], coarse[ind-cw], coarse[ind-cw+1]);
    //if(verbose) printf("    %f %f %f\n", coarse[ind-1], coarse[ind], coarse[ind+1]);
    //if(verbose) printf("    %f %f %f\n", coarse[ind+cw-1], coarse[ind+cw], coarse[ind+cw+1]);
    return 4./256. * (
        6.0f * (coarse(i2, j2-1) + coarse(i2-1, j2) + 6.0f*coarse(i2, j2) + coarse(i2+1, j2) + coarse(i2, j2+1))
            + coarse(i2-1, j2-1) + coarse(i2+1, j2-1) + coarse(i2-1, j2+1) + coarse(i2+1, j2+1));
  case 1: // i is odd, 2x3 stencil
    return 4./256. * (
        24.0f*(coarse(i2, j2) + coarse(i2+1, j2)) +
        4.0f*(coarse(i2, j2-1) + coarse(i2+1, j2-1) + coarse(i2, j2+1) + coarse(i2+1, j2+1)));
  case 2: // j is odd, 3x2 stencil
    return 4./256. * (
        24.0f*(coarse(i2, j2) + coarse(i2, j2+1)) +
        4.0f*(coarse(i2-1, j2) + coarse(i2-1, j2+1) + coarse(i2+1, j2) + coarse(i2+1, j2+1)));
  default: // case 3: // both are odd, 2x2 stencil
    return .25f * (coarse(i2, j2) + coarse(i2+1, j2) + coarse(i2, j2+1) + coarse(i2+1, j2+1));
  }
}



// helper to fill in one pixel boundary by copying it
static inline void ll_fill_boundary1(
    float *const input,
    const int wd,
    const int ht)
{
  // Fill an outer border of 1 pixel with inner pixel values.
  // copy second pixel column into first one
  for(int j=1;j<ht-1;j++) input[j*wd] = input[j*wd+1];
  // copy last-but-one pixel colukn into last one
  for(int j=1;j<ht-1;j++) input[j*wd+wd-1] = input[j*wd+wd-2];
  // copy second pixel row into first one
  memcpy(input,    input+wd, sizeof(float)*wd);
  // copy last-but-one pixel row into last one
  memcpy(input+wd*(ht-1), input+wd*(ht-2), sizeof(float)*wd);
}


// helper to fill in one pixel boundary by copying it
static inline void ll_fill_boundary2(
    Image& img)
{
  float* input = img.buf;
  int wd = img.pwidth;
  int ht = img.pheight;
  // Fill an outer border of 1 pixel with inner pixel values.
  // copy second pixel column into first one
  for(int j=2;j<ht-2;j++) input[j*wd] = input[j*wd+1] = input[j*wd+2];
  // copy last-but-one pixel colukn into last one
  for(int j=1;j<ht-1;j++) input[j*wd+wd-1] = input[j*wd+wd-2] = input[j*wd+wd-3];
  // copy second pixel row into first one
  memcpy(input,    input+wd*2, sizeof(float)*wd);
  memcpy(input+wd,    input+wd*2, sizeof(float)*wd);
  // copy last-but-one pixel row into last one
  memcpy(input+wd*(ht-2), input+wd*(ht-3), sizeof(float)*wd);
  memcpy(input+wd*(ht-1), input+wd*(ht-3), sizeof(float)*wd);
}

// helper to fill in two pixels boundary by copying it
static inline void ll_fill_boundary2(
    float *const input,
    const int wd,
    const int ht)
{
  for(int j=1;j<ht-1;j++) input[j*wd] = input[j*wd+1];
  if(wd & 1) for(int j=1;j<ht-1;j++) input[j*wd+wd-1] = input[j*wd+wd-2];
  else       for(int j=1;j<ht-1;j++) input[j*wd+wd-1] = input[j*wd+wd-2] = input[j*wd+wd-3];
  memcpy(input, input+wd, sizeof(float)*wd);
  if(!(ht & 1)) memcpy(input+wd*(ht-2), input+wd*(ht-3), sizeof(float)*wd);
  memcpy(input+wd*(ht-1), input+wd*(ht-2), sizeof(float)*wd);
}

static inline void gauss_expand(
    const float *const input, // coarse input
    float *const fine,        // upsampled, blurry output
    const int wd,             // fine res
    const int ht)
{
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static) collapse(2)
#endif
  for(int j=1;j<((ht-1)&~1);j++)  // even ht: two px boundary. odd ht: one px.
    for(int i=1;i<((wd-1)&~1);i++)
      fine[j*wd+i] = ll_expand_gaussian(input, i, j, wd, ht);
  ll_fill_boundary2(fine, wd, ht);
}

static inline void gauss_expand(
    Image& input, // coarse input
    Image& fine)  // upsampled, blurry output
{
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static) collapse(2)
#endif
  const int fw = fine.valid.width, fh = fine.valid.height;
  const int fl = fine.valid.left, fr = fine.valid.right;
  const int ft = fine.valid.top,  fb = fine.valid.bottom;

  for(int j = ft; j <= fb; j++)
    for(int i = fl; i <= fr; i++) {
      float val = ll_expand_gaussian(input, i, j);
      fine.set_pixel(i, j, val);
    }
  //ll_fill_boundary2(fine, wd, ht);
}



#if defined(__SSE2__)
static inline void gauss_reduce_sse2(
    const float *const input, // fine input buffer
    float *const coarse,      // coarse scale, blurred input buf
    const int wd,             // fine res
    const int ht)
{
  // blur, store only coarse res
  const int cw = (wd-1)/2+1, ch = (ht-1)/2+1;

  // this version is inspired by opencv's pyrDown_ :
  // - allocate 5 rows of ring buffer (aligned)
  // - for coarse res y
  //   - fill 5 coarse-res row buffers with 1 4 6 4 1 weights (reuse some from last time)
  //   - do vertical convolution via sse and write to coarse output buf

  const int stride = ((cw+8)&~7); // assure sse alignment of rows
  float *ringbuf = dt_alloc_align(64, sizeof(*ringbuf)*stride*5);
  float *rows[5] = {0};
  int rowj = 0; // we initialised this many rows so far

  for(int j=1;j<ch-1;j++)
  {
    // horizontal pass, convolve with 1 4 6 4 1 kernel and decimate
    for(;rowj<=2*j+2;rowj++)
    {
      float *const row = ringbuf + (rowj % 5)*stride;
      const float *const in = input + rowj*wd;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none)
#endif
      for(int i=1;i<cw-1;i++)
        row[i] = 6*in[2*i] + 4*(in[2*i-1]+in[2*i+1]) + in[2*i-2] + in[2*i+2];
    }

    // init row pointers
    for(int k=0;k<5;k++)
      rows[k] = ringbuf + ((2*j-2+k)%5)*stride;

    // vertical pass, convolve and decimate using SIMD:
    // note that we're ignoring the (1..cw-1) buffer limit, we'll pull in
    // garbage and fix it later by border filling.
    float *const out = coarse + j*cw;
    const float *const row0 = rows[0], *const row1 = rows[1],
        *const row2 = rows[2], *const row3 = rows[3], *const row4 = rows[4];
    const __m128 four = _mm_set1_ps(4.f), scale = _mm_set1_ps(1.f/256.f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none)
#endif
    for(int i=0;i<=cw-8;i+=8)
    {
      __m128 r0, r1, r2, r3, r4, t0, t1;
      r0 = _mm_load_ps(row0 + i);
      r1 = _mm_load_ps(row1 + i);
      r2 = _mm_load_ps(row2 + i);
      r3 = _mm_load_ps(row3 + i);
      r4 = _mm_load_ps(row4 + i);
      r0 = _mm_add_ps(r0, r4);
      r1 = _mm_add_ps(_mm_add_ps(r1, r3), r2);
      r0 = _mm_add_ps(r0, _mm_add_ps(r2, r2));
      t0 = _mm_add_ps(r0, _mm_mul_ps(r1, four));

      r0 = _mm_load_ps(row0 + i + 4);
      r1 = _mm_load_ps(row1 + i + 4);
      r2 = _mm_load_ps(row2 + i + 4);
      r3 = _mm_load_ps(row3 + i + 4);
      r4 = _mm_load_ps(row4 + i + 4);
      r0 = _mm_add_ps(r0, r4);
      r1 = _mm_add_ps(_mm_add_ps(r1, r3), r2);
      r0 = _mm_add_ps(r0, _mm_add_ps(r2, r2));
      t1 = _mm_add_ps(r0, _mm_mul_ps(r1, four));

      t0 = _mm_mul_ps(t0, scale);
      t1 = _mm_mul_ps(t1, scale);

      _mm_storeu_ps(out + i, t0);
      _mm_storeu_ps(out + i + 4, t1);
    }
    // process the rest
    for(int i=cw&~7;i<cw-1;i++)
      out[i] = (6*row2[i] + 4*(row1[i] + row3[i]) + row0[i] + row4[i])*(1.0f/256.0f);
  }
  dt_free_align(ringbuf);
  ll_fill_boundary1(coarse, cw, ch);
}
#endif

static inline void gauss_reduce(
    Image& input, // fine input buffer
    Image& coarse)      // coarse scale, blurred output buf
{
  // blur, store only coarse res
  const int cw = coarse.valid.width, ch = coarse.valid.height;
  const int cl = coarse.valid.left, cr = coarse.valid.right;
  const int ct = coarse.valid.top,  cb = coarse.valid.bottom;

  // this is the scalar (non-simd) code:
  const float a = 0.4f;
  const float w[5] = {1.f/4.f-a/2.f, 1.f/4.f, a, 1.f/4.f, 1.f/4.f-a/2.f};
  coarse.zero();
  // direct 5x5 stencil only on required pixels. Padding is 2 pixels.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) collapse(2)
#endif
  for(int j=ct;j<=cb;j++) for(int i=cl;i<=cr;i++)
    for(int jj=-2;jj<=2;jj++) for(int ii=-2;ii<=2;ii++) {
      coarse.add_to_pixel(i, j, input.get_pixel(2*i+ii, 2*j+jj) * w[ii+2] * w[jj+2]);
      if(false && i==cl) {
        printf("[gauss_reduce] i=%d j=%d  I=%d J=%d input=%f  W=%f  output=%f",
          i, j, 2*i+ii, 2*j+jj, input.get_pixel(2*i+ii, 2*j+jj), w[ii+2] * w[jj+2], coarse.get_pixel(i, j));
      if(ii==0 && jj==0) printf("  ***");
      printf("\n");
      }
    }
  //ll_fill_boundary2(coarse);
}



// allocate output buffer with monochrome brightness channel from input, padded
// up by padding on all four sides, dimensions written to wd2 ht2
static inline float *ll_pad_input(
    const float *const input,
    const int wd,
    const int ht,
    const int padding,
    int *wd2,
    int *ht2)
{
  const int stride = 1;
  *wd2 = 2*padding + wd;
  *ht2 = 2*padding + ht;
  float *const out = (float*)dt_alloc_align(64, *wd2**ht2*sizeof(*out));

  if( (*wd2) == wd && (*ht2) == ht ) {
    memcpy( out, input, *wd2**ht2*sizeof(*out) );
    return out;
  }

  { // pad by replication:
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(wd2, ht2)
#endif
    for(int j=0;j<ht;j++)
    {
      for(int i=0;i<padding;i++)
        out[(j+padding)**wd2+i] = input[stride*wd*j]; // L -> [0,1]
      for(int i=0;i<wd;i++)
        out[(j+padding)**wd2+i+padding] = input[stride*(wd*j+i)]; // L -> [0,1]
      for(int i=wd+padding;i<*wd2;i++)
        out[(j+padding)**wd2+i] = input[stride*(j*wd+wd-1)]; // L -> [0,1]
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(wd2, ht2)
#endif
    for(int j=0;j<padding;j++)
      memcpy(out + *wd2*j, out+padding**wd2, sizeof(float)**wd2);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) default(none) shared(wd2, ht2)
#endif
    for(int j=padding+ht;j<*ht2;j++)
      memcpy(out + *wd2*j, out + *wd2*(padding+ht-1), sizeof(float)**wd2);
  }
  return out;
}



static inline float ll_laplacian(
    const float *const coarse,   // coarse res gaussian
    const float *const fine,     // fine res gaussian
    const int i,                 // fine index
    const int j,
    const int wd,                // fine width
    const int ht,
    bool verbose=false)                // fine height
{
  const float c = ll_expand_gaussian(coarse,
      CLAMPS(i, 1, ((wd-1)&~1)-1), CLAMPS(j, 1, ((ht-1)&~1)-1), wd, ht);
  const float l = fine[j*wd+i] - c;
  if(verbose) printf("ll_laplacian(%d, %d): c=%f  fine=%f  laplacian=%f\n", i, j, c, fine[j*wd+i], l);
  return l;
}





#define max_levels 30
int pyramid_get_num_levels(int wd, int ht)
{
  //int nl = MIN(max_levels, 31-__builtin_clz(MAX(wd,ht))+1);
  int nl = 1, sz = MAX(wd,ht);
  while(sz > 1) {
    sz = (sz-1)/2+1;
    nl += 1;
  }
  return nl;
}


int pyramid_get_padding(int num_levels)
{
  int p = 1 << (num_levels);
  return p;
}



class GaussianPyramid
{
public:
  Image input;
  Image padded[max_levels];
  Image laplacian[max_levels];
  Image output[max_levels];
  float alpha[max_levels];
  float beta[max_levels];
  int width, height;
  int num_levels;
  int padding;
  bool verbose;
  float get_laplacian_coefficient(int x, int y, int l);
public:
  GaussianPyramid(float* input, LLRect& valid, LLRect& roi, int nl, bool add_padding=true, bool verbose=false);
  //~GaussianPyramid() {printf("Destroying gaussian pyramid\n");}

  void set_alpha(float al, float a0)
  {
    int last_level = num_levels-1;
    for(int l=last_level-1;l>=0;l--) {
      float i = static_cast<float>(l) / (num_levels-1);
      alpha[l] = al * i + a0 * (1.0f - i);
    }
  }
  void set_beta(float bl, float b0)
  {
    int last_level = num_levels-1;
    for(int l=last_level-1;l>=0;l--) {
      float i = static_cast<float>(l) / (num_levels-1);
      beta[l] = bl * i + b0 * (1.0f - i);
      printf("beta[%d] = %f\n", l, beta[l]);
    }
  }

  void fill_laplacian();
  void remap();
};


GaussianPyramid::GaussianPyramid(float* in, LLRect& valid, LLRect& roi, int nl, bool add_padding, bool v)
{
  char tstr[500];
  verbose = v;
  width = roi.width, height = roi.height;
  //input.pixels = (float*)in;
  //input.width = wd; input.height = ht;
  // don't divide by 2 more often than we can:
  num_levels = nl;
  int last_level = num_levels-1;
  padding = 2; //(add_padding) ? pyramid_get_padding(last_level) : 0;
  if(verbose) printf("GaussianPyramid: size=%dx%d levels=%d\n", valid.width, valid.height, num_levels);
  int w, h;
  //float* tbuf = ll_pad_input(in, wd, ht, padding, &w, &h);
  padded[0].set_buffer(valid, roi, in);
  sprintf(tstr,"padded[0]"); padded[0].set_name(tstr);
  if(verbose) printf("GaussianPyramid: level 0 filled, size=%dx%d, roi=%dx%d (%d -> %d)\n",
      padded[0].valid.width, padded[0].valid.height, padded[0].roi.width, padded[0].roi.height, padded[0].roi.left, padded[0].roi.right);
  if(verbose) padded[0].print();

  int xpadding[1000];
  int ypadding[1000];
  xpadding[last_level] = 1;
  ypadding[last_level] = 1;
  for(int l = (last_level-1); l > 0; l--) {
    xpadding[l] = xpadding[l+1] * 2 + 2;
    ypadding[l] = ypadding[l+1] * 2 + 2;
  }

  // allocate pyramid pointers for padded input
  for(int l = 1; l <= last_level; l++) {
    // input valid region scaled to level l
    int vleft = dl2(padded[0].valid.left, l);
    int vtop = dl2(padded[0].valid.top, l);
    int vwidth = dl(padded[0].valid.width, l);
    int vheight = dl(padded[0].valid.height, l);
    int vright = vleft + vwidth - 1;
    int vbottom = vtop + vheight - 1;
    // margins bayond which the pixel values are just repeated
    int mleft = 2;
    int mtop = 2;
    int mright = 3;
    int mbottom = 3;
    vleft -= mleft; vtop -= mtop; vright += mright; vbottom += mbottom;

    LLRect r, v;
    r.left = padded[l-1].roi.left / 2;
    r.width = dl(padded[0].roi.right - ul(r.left, l) + 1, l);
    r.top = padded[l-1].roi.top / 2;
    r.height = dl(padded[0].roi.bottom - ul(r.top, l) + 1, l);

    v.left = r.left - xpadding[l];
    v.top = r.top - ypadding[l];
    v.width = r.width + xpadding[l]*2 + 1;
    v.height = r.height + ypadding[l]*2 + 1;
    v.right = v.left +v.width - 1;
    v.bottom = v.top +v.height - 1;

    // remove redundant pixels at the image borders
    if(v.left < vleft) v.left = vleft;
    if(v.top < vtop) v.top = vtop;
    if(v.right > vright) v.right = vright;
    if(v.bottom > vbottom) v.bottom = vbottom;
    v.width = v.right - v.left + 1;
    v.height = v.bottom - v.top + 1;

    if(true && verbose) printf("GaussianPyramid: allocating level %d, size=%dx%d, roi=%dx%d\n",
        l, v.width, v.height, r.width, r.height);
    padded[l].alloc_buffer(v, r);
    if(verbose) printf("GaussianPyramid: left: %d -> %d, right: %d -> %d, width: %d -> %d\n",
        padded[l-1].roi.left, padded[l].roi.left, padded[l-1].roi.right, padded[l].roi.right, padded[l-1].roi.width, padded[l].roi.width);
    sprintf(tstr,"padded[%d]", l); padded[l].set_name(tstr);
    if(false && verbose) printf("GaussianPyramid: level %d allocated, size=%dx%d\n", l, padded[l].valid.width, padded[l].valid.height);
    if(false && verbose) padded[l].print();
  }
  //return;

  // allocate pyramid pointers for laplacian coefficients
  //for(int l=0;l<=last_level;l++) {
  //  laplacian[l].pixels = (float*)dt_alloc_align(64, sizeof(float)*dl(w,l)*dl(h,l));
  //  laplacian[l].width = dl(w,l); laplacian[l].height = dl(h,l);
  //}

  // allocate pyramid pointers for output
  //for(int l=0;l<=last_level;l++) {
  //  output[l].pixels = (float*)dt_alloc_align(64, sizeof(float)*dl(w,l)*dl(h,l));
  //  output[l].width = dl(w,l); output[l].height = dl(h,l);
  //}

  // create gauss pyramid of padded input, write coarse directly to output
#if defined(__SSE2__)
  if(use_sse2)
  {
    for(int l=1;l<last_level;l++)
      gauss_reduce_sse2(padded[l-1], padded[l], dl(w,l-1), dl(h,l-1));
    gauss_reduce_sse2(padded[last_level-1], output[last_level], dl(w,last_level-1), dl(h,last_level-1));
  }
  else
#endif
  {
    for(int l=1;l<=last_level;l++) {
      gauss_reduce(padded[l-1], padded[l]);
      if(verbose) printf("GaussianPyramid: level %d filled, size=%dx%d\n", l, padded[l].valid.width, padded[l].valid.height);
      if(verbose) padded[l].print();
    }
    //gauss_reduce(padded[last_level-1].pixels, padded[last_level].pixels, dl(w,last_level-1), dl(h,last_level-1));
    //if(verbose) printf("GaussianPyramid: level %d filled, size=%dx%d\n", last_level, dl(w,last_level), dl(h,last_level));
  }
  //if(verbose) padded[last_level].print();
}



void GaussianPyramid::fill_laplacian()
{
  int last_level = num_levels-1;
  if(verbose) printf("\n\n================\n\n");

  for(int l=last_level;l>0;l--) {
    gauss_expand(padded[l].pixels, output[l-1].pixels, output[l-1].width, output[l-1].height);
    if(verbose) printf("GaussianPyramid: level %d expanded, size=%dx%d\n", l, output[l-1].width, output[l-1].height);
    if(verbose) output[l-1].print();
  }

  if(verbose) printf("\n\n================\n\n");

  for(int l=last_level-1;l>=0;l--) {
    padded[l].copy(laplacian[l]);
    laplacian[l].add_image(output[l], -1);
    if(verbose) printf("GaussianPyramid: laplacian level %d created, size=%dx%d\n", l, laplacian[l].width, laplacian[l].height);
    if(verbose) laplacian[l].print();
    laplacian[l].zero();
  }

  //float test = ll_expand_gaussian( padded[2].pixels, 1, 2,
  //    laplacian[1].width, laplacian[1].height);
  //printf("ll_expand_gaussian(%d, %d, %d): %f\n", 2, 1, 2, test);
}


float GaussianPyramid::get_laplacian_coefficient(int x, int y, int l)
{
  // pixel coordinates in level l+1
  int xlp1 = x / 2, ylp1 = y / 2;

  // corner pixel coordinates in level 0
  int x0 = ul(xlp1, l+1), y0 = ul(ylp1, l+1);
  // size of the pixel region in level 0
  int w0 = ul(1, l+1), h0 = ul(1, l+1);

  int xpadding[1000];
  int ypadding[1000];
  xpadding[l+1] = 1;
  ypadding[l+1] = 1;
  for(int i = l; i >= 0; i--) {
    xpadding[i] = xpadding[i+1] * 2 + 2;
    ypadding[i] = ypadding[i+1] * 2 + 2;
  }

  if(verbose) {
    printf("Laplacian coefficient coordinates in level %d: %d,%d\n", l+1, xlp1, ylp1);
    printf("Laplacian coefficient coordinates in level %d: %d,%d\n", l, x, y);
    printf("Laplacian coefficient coordinates in level 0: %d,%d\n", x0, y0);
    printf("RoI size in level 0: %dx%d\n", w0, h0);
    printf("Paddings:\n");
    for(int i = 0; i <= l+1; i++) {
      printf("  l=%d   padding: %d,%d\n", i, xpadding[i], ypadding[i]);
    }
  }

  LLRect roi;
  roi.left = x0;// - xpadding[0];
  roi.top  = y0;// - ypadding[0];
  roi.width  = w0;// + xpadding[0] * 2;
  roi.height = h0;// + ypadding[0] * 2;

  LLRect crop;
  crop.left = x0 - xpadding[0];
  crop.top  = y0 - ypadding[0];
  crop.width  = w0 + xpadding[0] * 2;
  crop.height = h0 + ypadding[0] * 2;

  // gaussian coefficient for remapping
  float g0 = padded[l+1].get_pixel(xlp1, ylp1);
  float g1 = ll_expand_gaussian(padded[l+1], x, y, verbose);
  float g2 = padded[l].get_pixel(x, y);

  float g = g1;
  if(verbose) printf("g0=%0.5f  g1=%0.5f  g2=%0.5f  g=%0.5f\n", g0, g1, g2, g);

  Image cropped;
  padded[0].crop(crop, roi, cropped);
  if(verbose) { printf("cropped level 0:\n"); cropped.print(); }

  // remap pixel values in level 0
  float thr = THRESHOLD;
  for(int y = cropped.valid.top; y <= cropped.valid.bottom; y++) {
    for(int x = cropped.valid.left; x <= cropped.valid.right; x++) {
      float val = cropped(x, y);
      float delta = val - g;
      if( delta > thr ) {
        float diff = val - (g+thr);
        cropped.set_pixel(x, y, g+thr+(diff*beta[l]));
      } else if( delta < -thr ) {
        float diff = (g-thr) - val;
        cropped.set_pixel(x, y, g-thr-(diff*beta[l]));
      } else {
        if(verbose) printf("cropped(%d, %d): %+0.5f --> %+0.5f  delta=%+0.5f  \n", x, y, cropped(x, y), g+(delta*alpha[l]), delta);
        cropped.set_pixel(x, y, g+(delta*alpha[l]));
      }
    }
  }

  GaussianPyramid gp(cropped.buf, cropped.valid, roi, l+2, true, verbose);
  float c = ll_expand_gaussian(gp.padded[l+1], x, y, verbose);
  float f = gp.padded[l](x, y);
  float result = f - c;

  if(verbose) printf("(%d,%d): coarse=%0.5f  fine=%0.5f  laplacian=%0.5f\n", x, y, c, f, result);

  return result;
}


void GaussianPyramid::remap()
{
  int last_level = num_levels-1;
  if(verbose) printf("\n\n================\n\n");

  int w, h, p;
  padded[last_level].copy(output[last_level]);

  for(int l=last_level-1;l>=0;l--) {

    if(verbose) printf("\n------------------------\nremap: processing of level %d\n", l);
    else printf("remap: processing of level %d/%d\n", l, last_level-1);

    LLRect roi = padded[l].roi;
    LLRect valid;
    valid.left = roi.left - 1;
    valid.top = roi.top - 1;
    valid.width = roi.width + 3;
    valid.height = roi.height + 3;
    output[l].alloc_buffer(valid, roi);

    gauss_expand(output[l+1], output[l]);
    if(verbose) printf("remap: output[%d]:\n", l+1);
    if(verbose) output[l+1].print();
    if(verbose) printf("remap: output[%d] expanded:\n", l+1);
    if(verbose) output[l].print();

    //getchar();
    //continue;

    float thr = THRESHOLD;
    // pixel coordinates in laplacian level l
    for(int lpy = output[l].valid.top; lpy <= output[l].valid.bottom; lpy++) {
    //for(int lpy = (output[l].valid.top+output[l].valid.height/2); lpy <= (output[l].valid.top+output[l].valid.height/2); lpy++) {
      {
        int step = (padded[l].height-(padding*2))/10;
        //printf("lpy=%d  step=%d\n", lpy, step);
        if( (step>0) && ((lpy+1)%step) == 0 ) {printf("*"); fflush(stdout);}
      }

      for(int lpx = output[l].valid.left; lpx <= output[l].valid.right; lpx++) {
      //for(int lpx = (output[l].valid.left+output[l].valid.width/2); lpx <= (output[l].valid.left+output[l].valid.width/2); lpx++) {

        //bool verbose2 = true && (verbose && (lpy == (output[l].valid.top+output[l].valid.height/2)) && (lpx == (output[l].valid.left+output[l].valid.width/2)));
        bool verbose2 = true && (verbose && (lpy == (output[l].valid.top+output[l].valid.height/2)) && (lpx == 16));
        if(verbose2) printf("pixel coordinates in laplacian level %d: %d,%d\n", l, lpx, lpy);

        bool vtemp = verbose; verbose = verbose2;
        float laplacian = get_laplacian_coefficient(lpx, lpy, l);
        verbose = vtemp;
        if(verbose2) printf("remap: adding %+0.5f to output[%d](%d, %d)=%+0.5f\n", laplacian, l, lpx, lpy, output[l](lpx, lpy));
        output[l].add_to_pixel(lpx, lpy, laplacian);

        //if(true && verbose && (lpy == (output[l].valid.top+output[l].valid.height/2))) printf("\n");
      }
      //break;
    }
    if(verbose) printf("remap: output[%d] after add:\n", l);
    if(verbose) output[l].print();
    printf("\n");
    //if(l<4) getchar();
    //if( l==0 ) break;

    //getchar();
  }
}



//#define TEST_IMG 1


int
main( int argc, char **argv )
{
  float compression = 2;
  float compression0 = 2;
  float contrast = 1.5;
  float contrast0 = 1.0;
  float thr = 0.5;
  size_t optind;
  std::string outfile = "/tmp/llfout.tif";
  for (optind = 1; optind < argc && argv[optind][0] == '-'; optind++) {
      switch (argv[optind][1]) {
      case 'c': optind++; compression = atof(argv[optind]); compression0 = compression; break;
      case 'C': optind++; contrast = atof(argv[optind]); optind++; contrast0 = atof(argv[optind]); break;
      case 't': optind++; thr = atof(argv[optind]); break;
      case 'o': optind++; outfile = argv[optind]; break;
      default:
          fprintf(stderr, "Usage: %s \n", argv[0]);
          exit(EXIT_FAILURE);
      }
  }

  ALPHA = contrast;
  ALPHA0 = contrast0;
  BETA = 1.0f / compression;
  BETA0 = 1.0f / compression0;
  THRESHOLD = thr;


#ifdef TEST_IMG
  int W = 32;
  int H = 1;
  int Wo, Ho;

  float* buf;
  float* input = (float*)malloc(sizeof(float)*W*H);

  for(int i = 0; i < W; i++) {
    input[i] = 0.1;
  }
  input[W/2] = 1.;

  LLRect v;
  v.left = 0;
  v.top = 0;
  v.width = W;
  v.height = H;

  LLRect r = v;
  int border = 0;
  //W = 16;
  r.left = border;
  r.top = 0;
  r.width = W;
  r.height = H;

  bool verbose = true;

#else

  VipsImage *image;


  // Create VipsImage from given file
  image = vips_image_new_from_file( argv[argc-1], NULL );
  if( !image ) {
    printf("Failed to load \"%s\"\n",argv[argc-1]);
    return 1;
  }

  printf("image:            %p\n",image);
  printf("# of bands:       %d\n",image->Bands);
  printf("band format:      %d\n",image->BandFmt);
  printf("type:             %d\n",image->Type);
  printf("image dimensions: %d x %d\n",image->Xsize,image->Ysize);

  size_t array_sz;
  float* buf = (float*)vips_image_write_to_memory( image, &array_sz );
  if( !buf ) return 1;

  float log2 = log(2);
  float* logl = (float*)malloc(sizeof(float) * image->Xsize * image->Ysize);
  for(int y = 0; y < image->Ysize; y++) {
    for(int x = 0; x < image->Xsize; x++) {
      //printf("image dimensions: %d x %d\n",image->Xsize,image->Ysize);
      //std::cout<<"y="<<y<<"  x="<<x<<std::endl;
      float R = buf[(y*image->Xsize+x)*image->Bands];
      float G = buf[(y*image->Xsize+x)*image->Bands+1];
      float B = buf[(y*image->Xsize+x)*image->Bands+2];
      float L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
      if( L < 1.0e-15 ) L = 1.0e-15;
#ifdef LOGL
      logl[y*image->Xsize+x] = log(L) / log2;
#else
      logl[y*image->Xsize+x] = L;
#endif
    }
  }
  float* input = (float*)logl;

  int W = image->Xsize;
  int H = image->Ysize;

  LLRect v;
  v.left = 0;
  v.top = 0;
  v.width = W;
  v.height = H;

  LLRect r = v;

  bool verbose = false;
#endif

  int nl = pyramid_get_num_levels(r.width, r.height) + 0;

  GaussianPyramid gp(input, v, r, nl, true, verbose);
  gp.set_alpha(ALPHA, ALPHA0);
  gp.set_beta(BETA, BETA0);
  //getchar();
  //gp.get_laplacian_coefficient(2, 0, 2); getchar();
  //gp.fill_laplacian();
  printf("\n\n================================\n\n");
  gp.remap();
  printf("gp.output[0].roi=%dx%d\n", gp.output[0].roi.width, gp.output[0].roi.height);

#ifndef TEST_IMG

  for(int y = 0; y < image->Ysize; y++) {
    for(int x = 0; x < image->Xsize; x++) {
      float R = buf[(y*image->Xsize+x)*image->Bands];
      float G = buf[(y*image->Xsize+x)*image->Bands+1];
      float B = buf[(y*image->Xsize+x)*image->Bands+2];
      float L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
      float rlL = gp.output[0](x, y);
#ifdef LOGL
      float rL = STRENGTH*pow(2, rlL) + (1.0f-STRENGTH)*L;
#else
      float rL = STRENGTH*rlL + (1.0f-STRENGTH)*L;
#endif
      float ratio = (fabs(L) > 1.0e-15) ? rL / L : 1;
      //if( y == image->Ysize/2 ) printf("L=%f  rlL=%f  rL=%f ratio=%f\n", L, rlL, rL, ratio);
      R *= ratio;
      G *= ratio;
      B *= ratio;
      buf[(y*image->Xsize+x)*image->Bands] = R;
      buf[(y*image->Xsize+x)*image->Bands+1] = G;
      buf[(y*image->Xsize+x)*image->Bands+2] = B;
    }
  }

  printf("Saving output to %s\n", outfile.c_str());
  //VipsImage* out = vips_image_new_from_memory( outimg.pixels,
  //    sizeof(float)*outimg.width*outimg.height,
  //    outimg.width, outimg.height, 1, image->BandFmt );
  VipsImage* out = vips_image_new_from_memory( buf, sizeof(float) * image->Xsize * image->Ysize * image->Bands,
      image->Xsize, image->Ysize, image->Bands, image->BandFmt );
  vips_tiffsave( out, outfile.c_str(), "compression", VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE,
      "predictor", VIPS_FOREIGN_TIFF_PREDICTOR_NONE, NULL );

  FILE* fout = fopen((outfile+".txt").c_str(), "w");
  fprintf(fout, "compression: %0.2f %0.2f\ncontrast: %0.2f %0.2f\nthreshold: %0.2f\n", compression, compression0, contrast, contrast0, thr);
  fclose(fout);

  //LaplacianPyramid p( buf, array_sz, image->Xsize, image->Ysize, image->Bands, image->BandFmt, 2 );

  //p.remap();

  g_object_unref( image );
#endif

  return( 0 );
}
