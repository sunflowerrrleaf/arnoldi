#include <iostream>
#include <memory>
#include <cassert>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <chrono>

#define lapack_complex_double std::complex<double>
#define lapack_complex_double_real(z) (std::real(z))
#define lapack_complex_double_imag(z) (std::imag(z))

#include <cblas.h>
#include <lapacke.h>

const double COEF_DEG10_RE[] = {0.136112052334544905E-09, 0.963676398167865499E+01, -0.142343302081794718E+02, 0.513116990967461106E+01, -0.545173960592769901E+00, 0.115698077160221179E-01};
const double COEF_DEG10_IM[] = {0, -0.421091944767815675E+02, 0.176390663157379776E+02, -0.243277141223876469E+01, 0.284234540632477550E-01, 0.137170141788336280E-02};
const double ROOT_DEG10_RE[] = {-0.402773246751880265E+01, -0.328375288323169911E+01, -0.171540601576881357E+01, 0.894404701609481378E+00, 0.516119127202031791E+01};
const double ROOT_DEG10_IM[] = {0.119385606645509767E+01, 0.359438677235566217E+01, 0.603893492548519361E+01, 0.858275689861307000E+01, 0.113751562519165076E+02};

const double COEF_DEG14_RE[] = {0.183216998528140087E-11, 0.557503973136501826E+02, -0.938666838877006739E+02, 0.469965415550370835E+02, -0.961424200626061065E+01, 0.752722063978321642E+00, -0.188781253158648576E-01, 0.143086431411801849E-03};
const double COEF_DEG14_IM[] = {0, -0.204295038779771857E+03, 0.912874896775456363E+02, -0.116167609985818103E+02, -0.264195613880262669E+01, 0.670367365566377770E+00, -0.343696176445802414E-01, 0.287221133228814096E-03};
const double ROOT_DEG14_RE[] = {-0.562314417475317895E+01, -0.508934679728216110E+01, -0.399337136365302569E+01, -0.226978543095856366E+01, 0.208756929753827868E+00, 0.370327340957595652E+01, 0.889777151877331107E+01};
const double ROOT_DEG14_IM[] = {0.119406921611247440E+01, 0.358882439228376881E+01, 0.600483209099604664E+01, 0.846173881758693369E+01, 0.109912615662209418E+02, 0.136563731924991884E+02, 0.166309842834712071E+02};

const int N_ROOT = 5;
const int N_COEF = 6;

const double *ROOT_RE = ROOT_DEG10_RE;
const double *ROOT_IM = ROOT_DEG10_IM;
const double *COEF_RE = COEF_DEG10_RE;
const double *COEF_IM = COEF_DEG10_IM;

#define MIDX(A, r, c, lda) ((A)[(r) * (lda) + (c)])

void print_dm(const double *A, size_t n_row, size_t n_col, size_t lda, const std::string &name) {
  std::cout << "Matrix: " << name << "(" << n_row << " x " << n_col << ")" << std::endl;
  for (size_t r = 0; r < n_row; r++) {
    std::cout << "[";
    for (size_t c = 0; c < n_col; c++) {
      std::cout << MIDX(A, r, c, lda);
      if (c < n_col - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;
  }
}

void print_dv(const double *v, size_t n, const std::string &name) {
  std::cout << "Vector: " << name << "(" << n << ")" << std::endl;
  std::cout << "[";
  for (size_t i = 0; i < n; i++) {
    std::cout << v[i];
    if (i < n - 1) std::cout << " ";
  }
  std::cout << "]" << std::endl;
}

class ArnoldiExpmMultiply {
public:
  ArnoldiExpmMultiply(size_t n, int m) :
      _n(n), _m(m),
      _A(static_cast<double *>(malloc(sizeof(double) * n * n))),
      _v(static_cast<double *>(malloc(sizeof(double) * n))),
      _H(nullptr), _Q(nullptr), _eHm(nullptr), _eA(nullptr),
      _X_1(nullptr), _X_2(nullptr), _N(nullptr), _D(nullptr),
      _ipiv(static_cast<int *>(malloc(sizeof(int) * n))),
      _ipiv_m(static_cast<int *>(malloc(sizeof(int) * m))),
      _e1(nullptr), _eHme1(nullptr), _Hm_minus_theta_I_c(nullptr),
      _y_c(nullptr), _e1_c(nullptr) {}

  ~ArnoldiExpmMultiply() {
    free(_A);
    free(_v);
    if (_H) free(_H);
    if (_Q) free(_Q);
    if (_eAv) free(_eAv);
    if (_X_1) free(_X_1);
    if (_X_2) free(_X_2);
    if (_N) free(_N);
    if (_D) free(_D);
    if (_ipiv) free(_ipiv);
    if (_ipiv_m) free(_ipiv_m);
    if (_e1) free(_e1);
    if (_eHme1) free(_eHme1);
    if (_Hm_minus_theta_I_c) free(_Hm_minus_theta_I_c);
    if (_y_c) free(_y_c);
    if (_e1_c) free(_e1_c);
  }

  double *get_A() { return _A; }

  double *get_v() { return _v; }

  double *get_eAv() { return _eAv; }

private:
  const size_t _n;
  const int _m;

  double *_A;
  double *_v;

  double *_H;
  double *_Q;

  double *_eHm;
  double *_eAv;

  double *_eA;
  double *_X_1;
  double *_X_2;
  double *_N;
  double *_D;

  double *_e1;
  double *_eHme1;

  lapack_complex_double *_Hm_minus_theta_I_c;
  lapack_complex_double *_y_c;
  lapack_complex_double *_e1_c;

  int *_ipiv;
  int *_ipiv_m;

public:
  void prepare_pade() {
    _eAv = (double *) malloc(_n * sizeof(double));
    _X_1 = (double *) malloc(_n * _n * sizeof(double));
    _X_2 = (double *) malloc(_n * _n * sizeof(double));
    _N = (double *) malloc(_n * _n * sizeof(double));
    _D = (double *) malloc(_n * _n * sizeof(double));
    pade_ss_expm(_A, _n, _eA);
  }

  void pade_ss_expm_multiply() {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0, _eA, _n, _v, 1, 0.0, _eAv, 1);
  }

  void prepare_arnoldi_pade_ss() {
    _e1 = (double *) malloc(_m * sizeof(double));
    _eHme1 = (double *) malloc(_m * sizeof(double));
    _eAv = (double *) malloc(_n * sizeof(double));
    _H = (double *) malloc((_m + 1) * _m * sizeof(double));
    _Q = (double *) malloc(_n * (_m + 1) * sizeof(double));
    _X_1 = (double *) malloc(_m * _m * sizeof(double));
    _X_2 = (double *) malloc(_m * _m * sizeof(double));
    _N = (double *) malloc(_m * _m * sizeof(double));
    _D = (double *) malloc(_m * _m * sizeof(double));
    std::fill(_e1, _e1 + _m, 0.0);
    _e1[0] = 1.0;
  }

  void arnoldi_pade_ss_expm_multiply() {
    arnoldi_iteration();
    pade_ss_expm(_H, _m, _eHm);
    const double beta = cblas_dnrm2(_n, _v, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, _m, _m, 1.0, _eHm, _m, _e1, 1, 0.0, _eHme1, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, _n, _m, beta, _Q, _m + 1, _eHme1, 1, 0.0, _eAv, 1);
  }

  void prepare_arnoldi_chebyshev() {
    _e1_c = (lapack_complex_double *) malloc(_m * sizeof(lapack_complex_double));
    _eHme1 = (double *) malloc(_m * sizeof(double));
    _H = (double *) malloc((_m + 1) * _m * sizeof(double));
    _Q = (double *) malloc(_n * (_m + 1) * sizeof(double));
    _Hm_minus_theta_I_c = (lapack_complex_double *) malloc(_m * _m * sizeof(lapack_complex_double) * N_ROOT);
    _y_c = (lapack_complex_double *) malloc(_m * sizeof(lapack_complex_double) * N_ROOT);
    _eAv = (double *) malloc(_n * sizeof(double));
    _e1 = (double *) malloc(_m * sizeof(double));

    std::fill(_e1_c, _e1_c + _m, 0.0);
    _e1_c[0] = 1.0;
  }

  void arnoldi_chebyshev_expm_multiply() {
    arnoldi_iteration();
    cblas_daxpby(_m, COEF_RE[0], _e1, 1, 0.0, _eHme1, 1);

    const size_t m_sqrd = _m * _m;

#pragma omp parallel for default(none) shared(N_ROOT, ROOT_RE, ROOT_IM, COEF_RE, COEF_IM, m_sqrd) num_threads(8)
    for (int i = 0; i < N_ROOT; i++) {
      for (size_t j = 0; j < m_sqrd; j++) _Hm_minus_theta_I_c[m_sqrd * i + j] = -_H[j];

      lapack_complex_double theta(ROOT_RE[i], ROOT_IM[i]);
      for (size_t j = 0; j < _m; j++) {
        _Hm_minus_theta_I_c[m_sqrd * i + j * _m + j] -= theta;
      }

      cblas_zcopy(_m, _e1_c, 1, _y_c + _m * i, 1);
      LAPACKE_zgesv(LAPACK_ROW_MAJOR, _m, 1, _Hm_minus_theta_I_c + m_sqrd * i, _m,
                    _ipiv_m, _y_c + _m * i, 1);

      lapack_complex_double alpha(COEF_RE[i + 1], COEF_IM[i + 1]);
      cblas_zscal(_m, &alpha, _y_c + _m * i, 1);

#pragma omp critical
      {
        for (size_t j = 0; j < _m; j++) _eHme1[j] += lapack_complex_double_real(_y_c[_m * i + j]);
      }
    }

    // the same as with arnoldi pade ss
    const double beta = cblas_dnrm2(_n, _v, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, _n, _m, beta, _Q, _m
                                                               + 1, _eHme1, 1, 0.0, _eAv, 1);
  }

private:

  void pade_ss_expm(double *A, size_t n, double *&eA) {
    int ret = 0;
    int rows = n;
    int leading_dim = n;

    assert(rows > 0);

    // how to set t?
    const double t = 1.0;
    double inf_norm = LAPACKE_dlange(CblasRowMajor, 'I', rows, rows, A, leading_dim);
    assert(inf_norm > 0.0);
    int At_norm = static_cast<int>(inf_norm * t);
    int scale_exp = std::min(30, std::max(0, 1 + At_norm));

    double Ascal = t / std::pow(2.0, scale_exp);
    assert(Ascal > 0.0);
    cblas_dscal(n * n, Ascal, A, 1);

    constexpr int q = 3;
    double c = 0.5;
    double sign = -1.0;

    cblas_dcopy(n * n, A, 1, _X_1, 1);

    for (size_t i = 0; i < n * n; i++) {
      _X_2[i] = 0.0;
      _N[i] = 0.0;
      _D[i] = 0.0;
    }

    for (size_t i = 0; i < static_cast<size_t>(rows); i++) {
      MIDX(_N, i, i, n) = 1.0;
      MIDX(_D, i, i, n) = 1.0;
    }

    cblas_daxpy(n * n, c, _X_1, 1, _N, 1);
    cblas_daxpy(n * n, sign * c, _X_1, 1, _D, 1);

    for (int i = 2; i <= q;) {
      c = c * (q - i + 1) / (i * (2 * q - i + 1));
      sign *= -1.0;

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, rows, rows,
                  1.0, A, leading_dim, _X_1, leading_dim, 0.0,
                  _X_2, leading_dim);
      cblas_daxpy(n * n, c, _X_2, 1, _N, 1);
      cblas_daxpy(n * n, sign * c, _X_2, 1, _D, 1);

      i += 1;

      if (i > q) { break; }

      c = c * (q - i + 1) / (i * (2 * q - i + 1));
      sign *= -1.0;

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, rows, rows,
                  1.0, A, leading_dim, _X_2, leading_dim, 0.0,
                  _X_1, leading_dim);
      cblas_daxpy(n * n, c, _X_1, 1, _N, 1);
      cblas_daxpy(n * n, sign * c, _X_1, 1, _D, 1);

      i += 1;
    }

    ret = LAPACKE_dgesv(CblasRowMajor, rows, rows, _D, leading_dim, _ipiv, _N, leading_dim);
    assert(ret == 0);

    auto r1 = _N;
    auto r2 = _D;
    for (int i = 0; i < scale_exp; ++i) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, rows, rows,
                  1.0, r1, leading_dim, r1, leading_dim, 0.0,
                  r2, leading_dim);
      std::swap(r1, r2);
    }

    eA = r1;
  }

  void arnoldi_iteration() {
    std::fill(_H, _H + (_m + 1) * _m, 0.0);
    std::fill(_Q, _Q + _n * (_m + 1), 0.0);

    // beta = ||v||_2 is also used in the final approximation
    const double beta = cblas_dnrm2(_n, _v, 1);

    // normalization: v1 = v / ||v||_2
    cblas_dcopy(_n, _v, 1, _Q, _m + 1);
    cblas_dscal(_n, 1.0 / beta, _Q, _m + 1);

    // arnoldi iterations
    for (int j = 0; j < _m; j++) {
      // candidate for the next basis: w = A^(j+1) * v
      cblas_dgemv(CblasRowMajor, CblasNoTrans, _n, _n, 1.0, _A, _n,
                  _Q + j, _m + 1, 0.0, _Q + j + 1, _m + 1);

      // remove components in directions of other bases
      for (int i = 0; i <= j; i++) {
        const double hij = cblas_ddot(_n, _Q + i, _m + 1, _Q + j + 1, _m + 1);
        MIDX(_H, i, j, _m) = hij;
        cblas_daxpy(_n, -hij, _Q + i, _m + 1, _Q + j + 1, _m + 1);
      }

      // compute next basis
      const double h_jp1_j = cblas_dnrm2(_n, _Q + j + 1, _m + 1);
      if (h_jp1_j > 1e-12) {
        MIDX(_H, j + 1, j, _m) = h_jp1_j;
        cblas_dscal(_n, 1.0 / h_jp1_j, _Q + j + 1, _m + 1);
      } else {
        break;
      }
    }

    // print_dm(_H, _m + 1, _m, _m, "H");
    // print_dm(_Q, _n, _m + 1, _m + 1, "Q");
  }

};


int main(int argc, char *argv[]) {

//  openblas_set_num_threads(4);
  std::cout << "num of threads: " << openblas_get_num_threads() << std::endl;

  std::ifstream fin("lagrange-ng-regression/100taxa_6regions_1workers_1tpw/11/A.txt");
  int n;
  fin >> n;
  std::vector<double> data(n * n);
  for (int i = 0; i < n * n; i++) fin >> data[i];
  fin.close();

  std::vector<double> v_data(n);
  v_data[0] = 0.1915194503788923;
  v_data[1] = 0.6221087710398319;
  v_data[2] = 0.4377277390071145;
  v_data[3] = 0.7853585837137692;
  v_data[4] = 0.7799758081188035;
  v_data[5] = 0.2725926052826416;
  v_data[6] = 0.2764642551430967;
  v_data[7] = 0.8018721775350193;
  v_data[8] = 0.9581393536837052;
  v_data[9] = 0.8759326347420947;
  v_data[10] = 0.35781726995786667;
  v_data[11] = 0.5009951255234587;
  v_data[12] = 0.6834629351721363;
  v_data[13] = 0.7127020269829002;
  v_data[14] = 0.37025075479039493;
  v_data[15] = 0.5611961860656249;
  v_data[16] = 0.5030831653078097;
  v_data[17] = 0.013768449590682241;
  v_data[18] = 0.772826621612374;
  v_data[19] = 0.8826411906361166;
  v_data[20] = 0.3648859839013723;
  v_data[21] = 0.6153961784334937;
  v_data[22] = 0.07538124164297655;
  v_data[23] = 0.3688240060019745;
  v_data[24] = 0.9331401019825216;
  v_data[25] = 0.6513781432265774;
  v_data[26] = 0.3972025777261542;
  v_data[27] = 0.7887301429407455;
  v_data[28] = 0.31683612216887125;
  v_data[29] = 0.5680986526260692;
  v_data[30] = 0.8691273895612258;
  v_data[31] = 0.43617342389567937;
  v_data[32] = 0.8021476420801591;
  v_data[33] = 0.14376682451456457;
  v_data[34] = 0.7042609711183354;
  v_data[35] = 0.7045813081895725;
  v_data[36] = 0.21879210567408858;
  v_data[37] = 0.924867628615565;
  v_data[38] = 0.44214075540417663;
  v_data[39] = 0.9093159589724725;
  v_data[40] = 0.0598092227798519;
  v_data[41] = 0.18428708381381365;
  v_data[42] = 0.04735527880151513;
  v_data[43] = 0.6748809435823302;
  v_data[44] = 0.5946247799344488;
  v_data[45] = 0.5333101629987506;
  v_data[46] = 0.04332406269480349;
  v_data[47] = 0.5614330800633979;
  v_data[48] = 0.329668445620915;
  v_data[49] = 0.5029668331126184;
  v_data[50] = 0.11189431757440382;
  v_data[51] = 0.6071937062184846;
  v_data[52] = 0.5659446430505314;
  v_data[53] = 0.00676406199000279;
  v_data[54] = 0.617441708804297;
  v_data[55] = 0.9121228864331543;
  v_data[56] = 0.7905241330570334;
  v_data[57] = 0.9920814661883615;
  v_data[58] = 0.9588017621528665;
  v_data[59] = 0.7919641352916398;
  v_data[60] = 0.2852509600245098;
  v_data[61] = 0.624916705305911;
  v_data[62] = 0.47809379567067456;
  v_data[63] = 0.19567517866589823;

  int m = 12;
  int max_nv = 100;
  int ntest = 100;
  std::vector<long double> timings(max_nv);
  std::fill(timings.begin(), timings.end(), 0.0);

#define EXPM_METHOD 2
#define CHECK_RESULT 0
#define BREAK_AT_FIRST 0

  for (int test = 0; test < ntest; test++) {
    std::cout << "test: " << test << std::endl;

    for (int nv = 1; nv <= max_nv; nv++) {
      ArnoldiExpmMultiply expm_mul(n, m);
      auto A = expm_mul.get_A();
      auto v = expm_mul.get_v();
      std::copy(data.begin(), data.end(), A);
      std::copy(v_data.begin(), v_data.end(), v);

#if EXPM_METHOD == 0
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      expm_mul.prepare_pade();
      for (int i = 0; i < nv; i++) {
        expm_mul.pade_ss_expm_multiply();
      }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#elif EXPM_METHOD == 1
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      expm_mul.prepare_arnoldi_pade_ss();
      for (int i = 0; i < nv; i++) {
        expm_mul.arnoldi_pade_ss_expm_multiply();
      }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#elif EXPM_METHOD == 2
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      expm_mul.prepare_arnoldi_chebyshev();
      for (int i = 0; i < nv; i++) {
        expm_mul.arnoldi_chebyshev_expm_multiply();
      }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif

      timings[nv - 1] += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

#if CHECK_RESULT
      auto eAv = expm_mul.get_eAv();
      std::cout << "[";
      for (int i = 0; i < n; i++) {
        std::cout << eAv[i];
        if (i < n - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
#if BREAK_AT_FIRST
      break;
    }
    break;
#else
    }
#endif
#else
    }
#endif
  }

#if EXPM_METHOD == 0
  std::cout << "timings_pade_ss = [";
#elif EXPM_METHOD == 1
  std::cout << "timings_arnoldi_pade_ss = [";
#elif EXPM_METHOD == 2
  std::cout << "timings_arnoldi_chebyshev = [";
#endif
  for (size_t i = 0; i < timings.size(); i++) {
    std::cout << timings[i] / ((long double) ntest);
    if (i < timings.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  return 0;
}