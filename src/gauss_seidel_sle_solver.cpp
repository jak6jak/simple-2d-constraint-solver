#include "../include/gauss_seidel_sle_solver.h"

#include <cmath>
#include <assert.h>

// ARM64 NEON optimization for dot products
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define USE_NEON_DOUBLE 1
#endif

atg_scs::GaussSeidelSleSolver::GaussSeidelSleSolver()
    : atg_scs::SleSolver(true)
{
    m_maxIterations = 128;
    m_minDelta = 1E-1;

    m_M.initialize(1, 1);
}

atg_scs::GaussSeidelSleSolver::~GaussSeidelSleSolver() {
    m_M.destroy();
    m_reg.destroy();
}

bool atg_scs::GaussSeidelSleSolver::solve(
        SparseMatrix<3> &J,
        Matrix &W,
        Matrix &right,
        Matrix *previous,
        Matrix *result)
{
    const int n = right.getHeight();
    
    result->resize(1, n);

    if (previous != nullptr && previous->getHeight() == n) {
        result->set(previous);
    }

    J.rightScale(W, &m_reg);
    m_reg.multiplyTranspose(J, &m_M);

    for (int i = 0; i < m_maxIterations; ++i) {
        const double maxDelta = solveIteration(
                m_M,
                right,
                result,
                result);

        if (maxDelta < m_minDelta) {
            return true;
        }
    }

    return false;
}

bool atg_scs::GaussSeidelSleSolver::solveWithLimits(
    SparseMatrix<3> &J,
    Matrix &W,
    Matrix &right,
    Matrix &limits,
    Matrix *result,
    Matrix *previous)
{
    const int n = right.getHeight();
    if (result->getHeight() != n) {
        result->initialize(1, n);
    }

    if (previous != nullptr && previous->getHeight() == n) {
        result->set(previous);
    }

    J.rightScale(W, &m_reg);
    m_reg.multiplyTranspose(J, &m_M);

    for (int i = 0; i < m_maxIterations; ++i) {
        const double maxDelta = solveIteration(
            m_M,
            right,
            limits,
            result,
            result);

        if (maxDelta < m_minDelta) {
            return true;
        }
    }

    return false;
}

double atg_scs::GaussSeidelSleSolver::solveIteration(
        Matrix &left,
        Matrix &right,
        Matrix *k_next,
        Matrix *k)
{
    double maxDifference = 0.0;
    const int n = k->getHeight();

    for (int i = 0; i < n; ++i) {
        double s0 = 0.0, s1 = 0.0;
        
#if USE_NEON_DOUBLE
        const double* left_row = left.getRowPtr(i);
        const double* k_next_data = k_next->getData();
        const double* k_data = k->getData();
        
        // Vectorized sum for j < i
        int j = 0;
        float64x2_t sum0_vec = vdupq_n_f64(0.0);
        for (; j + 1 < i; j += 2) {
            float64x2_t left_vec = vld1q_f64(&left_row[j]);
            float64x2_t k_vec = vld1q_f64(&k_next_data[j]);
            sum0_vec = vfmaq_f64(sum0_vec, left_vec, k_vec);
        }
        s0 = vaddvq_f64(sum0_vec);
        for (; j < i; ++j) {
            s0 += left_row[j] * k_next_data[j];
        }
        
        // Vectorized sum for j > i  
        j = i + 1;
        float64x2_t sum1_vec = vdupq_n_f64(0.0);
        for (; j + 1 < n; j += 2) {
            float64x2_t left_vec = vld1q_f64(&left_row[j]);
            float64x2_t k_vec = vld1q_f64(&k_data[j]);
            sum1_vec = vfmaq_f64(sum1_vec, left_vec, k_vec);
        }
        s1 = vaddvq_f64(sum1_vec);
        for (; j < n; ++j) {
            s1 += left_row[j] * k_data[j];
        }
#else
        for (int j = 0; j < i; ++j) {
            s0 += left.get(j, i) * k_next->get(0, j);
        }

        for (int j = i + 1; j < n; ++j) {
            s1 += left.get(j, i) * k->get(0, j);
        }
#endif

        const double k_next_i =
            (1 / left.get(i, i)) * (right.get(0, i) - s0 - s1);

        const double min_k = std::fmax(1E-3, k->get(0, i));
        const double delta = (std::abs(k_next_i) - min_k) / min_k;
        maxDifference = (delta > maxDifference)
            ? delta
            : maxDifference;

        k_next->set(0, i, k_next_i);
    }

    return maxDifference;
}

double atg_scs::GaussSeidelSleSolver::solveIteration(
        Matrix &left,
        Matrix &right,
        Matrix &limits,
        Matrix *k_next,
        Matrix *k)
{
    double maxDifference = 0.0;
    const int n = k->getHeight();

    for (int i = 0; i < n; ++i) {
        double s0 = 0.0, s1 = 0.0;
        
#if USE_NEON_DOUBLE
        const double* left_row = left.getRowPtr(i);
        const double* k_next_data = k_next->getData();
        const double* k_data = k->getData();
        
        // Vectorized sum for j < i
        int j = 0;
        float64x2_t sum0_vec = vdupq_n_f64(0.0);
        for (; j + 1 < i; j += 2) {
            float64x2_t left_vec = vld1q_f64(&left_row[j]);
            float64x2_t k_vec = vld1q_f64(&k_next_data[j]);
            sum0_vec = vfmaq_f64(sum0_vec, left_vec, k_vec);
        }
        s0 = vaddvq_f64(sum0_vec);
        for (; j < i; ++j) {
            s0 += left_row[j] * k_next_data[j];
        }
        
        // Vectorized sum for j > i  
        j = i + 1;
        float64x2_t sum1_vec = vdupq_n_f64(0.0);
        for (; j + 1 < n; j += 2) {
            float64x2_t left_vec = vld1q_f64(&left_row[j]);
            float64x2_t k_vec = vld1q_f64(&k_data[j]);
            sum1_vec = vfmaq_f64(sum1_vec, left_vec, k_vec);
        }
        s1 = vaddvq_f64(sum1_vec);
        for (; j < n; ++j) {
            s1 += left_row[j] * k_data[j];
        }
#else
        for (int j = 0; j < i; ++j) {
            s0 += left.get(j, i) * k_next->get(0, j);
        }

        for (int j = i + 1; j < n; ++j) {
            s1 += left.get(j, i) * k->get(0, j);
        }
#endif

        const double k_next_i =
            (1 / left.get(i, i)) * (right.get(0, i) - s0 - s1);

        const double limitMin = limits.get(0, i);
        const double limitMax = limits.get(1, i);
        const double x = std::fmax(limitMin, std::fmin(limitMax, k_next_i));

        const double min_k = std::fmax(1E-3, std::abs(k->get(0, i)));
        const double delta = std::abs(x - k->get(0, i)) / min_k;
        maxDifference = (delta > maxDifference)
            ? delta
            : maxDifference;

        k_next->set(0, i, x);
    }

    return maxDifference;
}
