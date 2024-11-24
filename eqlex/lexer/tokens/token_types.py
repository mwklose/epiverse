from enum import Enum, auto

class TokenType(Enum):
    # Syntax Tokens
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    COMMA = ","
    DOT = "."

    # Binary operator Tokens
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"

    # TODO: semicolon?

    # Logical Operators
    BANG = "!"
    EQ = "="
    EQ_EQ = "=="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    ORBAR = "|"
    AMPERSAND = "&"

    # Literals
    IDENTIFIER = "ID"
    NUMBER = "NUM"

    # Distributions
    # TODO: add more distributions?
    BERN = "BERN"
    NORMAL = "NORM"
    BINOM = "BINOM"
    POISSON = "POISSON"
    NEGBIN = "NEGBINOM"
    WEIBULL = "WEIBULL"
    EXPON = "EXPON"

    # Known Functions
    FLOOR = "FLOOR"
    CEIL = "CEIL"
    LOG = "LOG"
    LOGIT = "LOGIT"
    EXP = "EXP"
    EXPIT = "EXPIT"
    SQRT = "SQRT"
    POW = "POW"
    
    # Keywords?
    

    # EOF
    EOF = "EOF"
