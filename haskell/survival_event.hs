module SurvivalEvent where 

data Event a = 
    Outcome a | Event a 
    deriving (Show, Eq)

instance (Ord a) => Ord (Event a) where 
    compare (Outcome x) (Outcome y) = compare x y
    compare (Censor x) (Censor y) = compare x y
    compare (Outcome x) (Censor y) 
        | x <= y    = LT
        | x > y     = GT
    compare (Censor x) (Outcome y) 
        | x < y     = LT
        | x >= y    = GT


getTime :: Event a -> a
getTime (Outcome x) = x
getTime (Censor x) = x

isOutcome :: Event a -> Bool
isOutcome (Outcome _) = True
isOutcome _ = False

isCensor :: Event a -> Bool
isCensor = not.isOutcome

-- TODO: this is harder than it seems