#ifndef ATG_SIMPLE_2D_CONSTRAINT_SOLVER_CHOLESKY_SLE_SOLVER_H
#define ATG_SIMPLE_2D_CONSTRAINT_SOLVER_CHOLESKY_SLE_SOLVER_H

#include "sle_solver.h"
#include "matrix.h"
#include "sparse_matrix.h"

namespace atg_scs {
    class CholeskySleSolver : public SleSolver {
        public:
            CholeskySleSolver();
            virtual ~CholeskySleSolver();

            virtual bool solve(
                SparseMatrix<3> &J,
                Matrix &W,
                Matrix &right,
                Matrix *previous,
                Matrix *result);

        protected:
            Matrix m_L;      // Lower triangular matrix from decomposition
            Matrix m_M;      // J * W * J^T
            SparseMatrix<3> m_reg;    // Temporary register for J * W
            Matrix m_y;      // Intermediate solution
    };
} /* namespace atg_scs */

#endif /* ATG_SIMPLE_2D_CONSTRAINT_SOLVER_CHOLESKY_SLE_SOLVER_H */
