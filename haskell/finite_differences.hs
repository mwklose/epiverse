data Decimal
finite_differences_1d :: (Decimal -> Decimal) -> Decimal -> Decimal
finite_differences_1d f x = let eps = 1e-9 in (f (x + eps) - f (x - eps)) / (2.0 * eps)