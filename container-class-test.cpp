

#include <string>
#include <cmath>
#include <cassert>
#include <vector>
#include <iterator>
#include <sys/time.h>


using byte = unsigned char;

struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


int main(int argc, char* argv [])
{
  if (argc != 3) {printf("USAGE: %s input_file_name error_bound\n\n", argv[0]); return -1;}

  // read input file (all "*size" variables in bytes)
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const size_t fsize = ftell(fin);
  if (fsize < sizeof(float)) {fprintf(stderr, "ERROR: input file is too small\n\n"); return -1;}
  if (fsize % sizeof(float) != 0) {fprintf(stderr, "ERROR: size of input file must be a multiple of %ld bytes\n", sizeof(float)); return -1;}
  const size_t num_elements = fsize / sizeof(float);
  float* input = new float [num_elements];
  fseek(fin, 0, SEEK_SET);
  const size_t insize = fread(input, 1, fsize, fin);
  fclose(fin);
  if (insize != fsize) {fprintf(stderr, "ERROR: could not read input file\n\n"); return -1;}

  // std::vector<float> g;
  // std::vector<float> m;
  // std::vector<float> v;
  // std::vector<float> p;

  // for (size_t i = 0; i < num_elements; i++) { g.push_back(input[i]); }
  // for (size_t i = 0; i < num_elements; i++) { m.push_back(input[i]); }
  // for (size_t i = 0; i < num_elements; i++) { v.push_back(input[i]); }
  // for (size_t i = 0; i < num_elements; i++) { p.push_back(input[i]); }

  float* g = new float [num_elements];
  float* m = new float [num_elements];
  float* v = new float [num_elements];
  float* p = new float [num_elements];

  for (size_t i = 0; i < num_elements; i++) { g[i] = input[i]; }
  for (size_t i = 0; i < num_elements; i++) { m[i] = input[i]; }
  for (size_t i = 0; i < num_elements; i++) { v[i] = input[i]; }
  for (size_t i = 0; i < num_elements; i++) { p[i] = input[i]; }

  const int timestep = 10000;
  const size_t i = 1;

  const float b1 = 0.1f;
  const float b2 = 0.1f;
  const float eps = 0.1f;
  const float grad_scale = 0.1f;
  const float step_size = 0.1f;
  const float decay = 0.1f;

  const int runs = 9;

  for (int r = 0; r < runs; r++)
{    CPUTimer timer;
    timer.start();
    for (size_t j = i; j < num_elements; j *= 2) {
      for (int t = 0; t < timestep; t++) {
        float scaled_grad = g[j]/grad_scale;
        m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
        v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
        float m_corrected = m[j] / (1.f-powf(b1, t));
        float v_corrected = v[j] / (1.f-powf(b2, t));
        float denom = sqrtf(v_corrected + eps);
        float update = (m_corrected/denom) + (decay*p[j]);
        p[j] = p[j] - (step_size*update);
      }
    }
    double runtime = timer.stop();
    printf("time: %.6f s\n", runtime);}

  delete [] input;

  return 0;
}
