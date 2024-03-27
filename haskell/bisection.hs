bisection :: (Double -> Double) ->  Double -> Double -> Int -> Double -> Double
bisection f lb ub iters tol
    | iters == 0        = error "Failed convergence in iterations"
    | (f(ub) - f(lb)) <= tol    = lb + (ub - lb) / 2
    | f ((lb+ub)/2) > 0 = bisection f lb ((lb+ub)/2) (iters - 1) tol
    | otherwise         = bisection f ((lb+ub)/2) ub (iters - 1) tol


