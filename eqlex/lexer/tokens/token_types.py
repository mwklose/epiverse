from enum import Enum, auto

class TokenType(Enum): 
    # Syntax Tokens
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    COMMA = auto()
    DOT = auto()

    # Binary operator Tokens
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()

    # TODO: semicolon?

    # Logical Operators
    BANG = auto()
    BANG_EQ = auto()
    EQ = auto()
    EQ_EQ = auto()
    GT = auto()
    GTE = auto()
    LT = auto()
    LTE = auto()
    ORBAR = auto()
    AMPERSAND = auto()

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    
    # Distributions
    # TODO: add more distributions?
    BERN = auto()
    NORMAL = auto()
    BINOM = auto()
    POISSON = auto()
    NEGBIN = auto()
    WEIBULL = auto()
    

    # Keywords?
    

    # EOF
    EOF = auto()
    ...