#include "../include/cholesky_sle_solver.h"

#include <cmath>
#include <assert.h>

atg_scs::CholeskySleSolver::CholeskySleSolver()
    : atg_scs::SleSolver(false)
{
    m_L.initialize(1, 1);
    m_M.initialize(1, 1);
    m_y.initialize(1, 1);
}

atg_scs::CholeskySleSolver::~CholeskySleSolver() {
    m_L.destroy();
    m_M.destroy();
    m_reg.destroy();
    m_y.destroy();
}

bool atg_scs::CholeskySleSolver::solve(
        SparseMatrix<3> &J,
        Matrix &W,
        Matrix &right,
        Matrix *previous,
        Matrix *result)
{
    // Compute M = J * W * J^T (symmetric positive semi-definite)
    J.rightScale(W, &m_reg);
    m_reg.multiplyTranspose(J, &m_M);

    const int n = m_M.getWidth();
    
    if (n == 0) return true;

    // Cholesky decomposition: M = L * L^T
    // L is lower triangular
    m_L.resize(n, n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = m_M.get(j, i);
            
            for (int k = 0; k < j; ++k) {
                sum -= m_L.get(k, i) * m_L.get(k, j);
            }
            
            if (i == j) {
                // Diagonal element
                if (sum <= 0.0) {
                    // Matrix is not positive definite, add regularization
                    sum = 1e-10;
                }
                m_L.set(j, i, std::sqrt(sum));
            } else {
                // Off-diagonal element
                const double L_jj = m_L.get(j, j);
                if (L_jj > 1e-12) {
                    m_L.set(j, i, sum / L_jj);
                } else {
                    m_L.set(j, i, 0.0);
                }
            }
        }
        // Upper triangle is zero
        for (int j = i + 1; j < n; ++j) {
            m_L.set(j, i, 0.0);
        }
    }

    // Solve L * y = b (forward substitution)
    m_y.resize(1, n);
    for (int i = 0; i < n; ++i) {
        double sum = right.get(0, i);
        for (int j = 0; j < i; ++j) {
            sum -= m_L.get(j, i) * m_y.get(0, j);
        }
        const double L_ii = m_L.get(i, i);
        m_y.set(0, i, L_ii > 1e-12 ? sum / L_ii : 0.0);
    }

    // Solve L^T * x = y (backward substitution)
    result->resize(1, n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = m_y.get(0, i);
        for (int j = i + 1; j < n; ++j) {
            sum -= m_L.get(i, j) * result->get(0, j);
        }
        const double L_ii = m_L.get(i, i);
        result->set(0, i, L_ii > 1e-12 ? sum / L_ii : 0.0);
    }

    // Check for NaN/Inf
    for (int i = 0; i < n; ++i) {
        if (std::isnan(result->get(0, i)) || std::isinf(result->get(0, i))) {
            return false;
        }
    }

    return true;
}
