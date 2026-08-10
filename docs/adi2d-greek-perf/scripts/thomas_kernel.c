/* Batched Thomas kernel — bit-identical to quantark's pure-NumPy
 * solve_tridiag_batch: same recurrences, same operation order, same 1e-14
 * pivot guard. Compile with -ffp-contract=off so no FMA contraction changes
 * the rounding sequence.
 *
 * Layout: all arrays C-contiguous, shape (n_sys, n), full-length convention
 * (sub[s][0] and sup[s][n-1] ignored). Returns 0 on success, 1 on pivot
 * failure (caller raises the same NumericalError as the NumPy version).
 */
#include <math.h>
#include <stddef.h>

int thomas_batch(const double *sub, const double *diag, const double *sup,
                 const double *rhs, double *x, double *cp, double *dp,
                 ptrdiff_t n_sys, ptrdiff_t n, double pivot_min)
{
    for (ptrdiff_t s = 0; s < n_sys; ++s) {
        const double *a = sub + s * n;
        const double *b = diag + s * n;
        const double *c = sup + s * n;
        const double *r = rhs + s * n;
        double *xx = x + s * n;
        if (fabs(b[0]) < pivot_min) return 1;
        cp[0] = c[0] / b[0];
        dp[0] = r[0] / b[0];
        for (ptrdiff_t i = 1; i < n; ++i) {
            double denom = b[i] - a[i] * cp[i - 1];
            if (fabs(denom) < pivot_min) return 1;
            cp[i] = c[i] / denom;
            dp[i] = (r[i] - a[i] * dp[i - 1]) / denom;
        }
        xx[n - 1] = dp[n - 1];
        for (ptrdiff_t i = n - 2; i >= 0; --i)
            xx[i] = dp[i] - cp[i] * xx[i + 1];
    }
    return 0;
}

/* Shared-coefficient variant for the ADI V-sweep (one operator, many RHS).
 * Eliminates once, back-substitutes per RHS; arithmetic per system is the
 * identical op sequence, so results stay bitwise equal to thomas_batch on
 * broadcast inputs. */
int thomas_multi_rhs(const double *a, const double *b, const double *c,
                     const double *rhs, double *x, double *cp, double *denoms,
                     ptrdiff_t n_rhs, ptrdiff_t n, double pivot_min)
{
    if (fabs(b[0]) < pivot_min) return 1;
    cp[0] = c[0] / b[0];
    denoms[0] = b[0];
    for (ptrdiff_t i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * cp[i - 1];
        if (fabs(denom) < pivot_min) return 1;
        cp[i] = c[i] / denom;
        denoms[i] = denom;
    }
    for (ptrdiff_t s = 0; s < n_rhs; ++s) {
        const double *r = rhs + s * n;
        double *xx = x + s * n;   /* dp stored in-place */
        xx[0] = r[0] / denoms[0];
        for (ptrdiff_t i = 1; i < n; ++i)
            xx[i] = (r[i] - a[i] * xx[i - 1]) / denoms[i];
        for (ptrdiff_t i = n - 2; i >= 0; --i)
            xx[i] = xx[i] - cp[i] * xx[i + 1];
    }
    return 0;
}
