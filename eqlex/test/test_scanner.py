from eqlex.lexer import Scanner, EqlexToken, TokenType


def test_one_plus_two(): 
    myscanner = Scanner("1+2")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="1", literal="1"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"), 
        EqlexToken(type=TokenType.NUMBER, lexeme="2", literal="2"), 
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    
    assert myscanner.token_list[0] == expected_tokens[0], f"Expected {expected_tokens[0]};\nReceived {myscanner.token_list[0]}"
    assert myscanner.token_list[1] == expected_tokens[1], f"Expected {expected_tokens[1]};\nReceived {myscanner.token_list[1]}"
    assert myscanner.token_list[2] == expected_tokens[2], f"Expected {expected_tokens[2]};\nReceived {myscanner.token_list[2]}"

    assert len(myscanner.token_list) == len(expected_tokens), "Expected number of tokens does not match actual."

def test_one_plus_two_spaces(): 
    myscanner = Scanner("1 + 2")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="1", literal="1"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"), 
        EqlexToken(type=TokenType.NUMBER, lexeme="2", literal="2"), 
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    
    assert myscanner.token_list[0] == expected_tokens[0], f"Expected {expected_tokens[0]};\nReceived {myscanner.token_list[0]}"
    assert myscanner.token_list[1] == expected_tokens[1], f"Expected {expected_tokens[1]};\nReceived {myscanner.token_list[1]}"
    assert myscanner.token_list[2] == expected_tokens[2], f"Expected {expected_tokens[2]};\nReceived {myscanner.token_list[2]}"

    assert len(myscanner.token_list) == len(expected_tokens), "Expected number of tokens does not match actual."


def test_one_plus_two_multiple_spaces(): 
    myscanner = Scanner("1  +   2")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="1", literal="1"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"), 
        EqlexToken(type=TokenType.NUMBER, lexeme="2", literal="2"), 
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    
    assert myscanner.token_list[0] == expected_tokens[0], f"Expected {expected_tokens[0]};\nReceived {myscanner.token_list[0]}"
    assert myscanner.token_list[1] == expected_tokens[1], f"Expected {expected_tokens[1]};\nReceived {myscanner.token_list[1]}"
    assert myscanner.token_list[2] == expected_tokens[2], f"Expected {expected_tokens[2]};\nReceived {myscanner.token_list[2]}"

    assert len(myscanner.token_list) == len(expected_tokens), "Expected number of tokens does not match actual."

def test_id_plus_number(): 
    idscanner_nospaces = Scanner("A+2")
    idscanner = Scanner("A + 2")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"), 
        EqlexToken(type=TokenType.NUMBER, lexeme="2", literal="2"), 
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner_nospaces.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner_nospaces.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"


def test_arithmetic(): 
    idscanner = Scanner("2 * A + 5 * (B - C) / 10")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="2", literal="2"), 
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"), 
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"), 
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"), 
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("), 
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="B", literal="B"),
        EqlexToken(type=TokenType.MINUS, lexeme="-", literal="-"), 
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="C", literal="C"), 
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"), 
        EqlexToken(type=TokenType.SLASH, lexeme="/", literal="/"),
        EqlexToken(type=TokenType.NUMBER, lexeme="10", literal="10"), 
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_arithmetic_with_negative(): 
    idscanner = Scanner("-2 * A + 5 * (B - C) / -10")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="-2", literal="-2"), 
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"), 
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"), 
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"), 
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("), 
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="B", literal="B"),
        EqlexToken(type=TokenType.MINUS, lexeme="-", literal="-"), 
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="C", literal="C"), 
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"), 
        EqlexToken(type=TokenType.SLASH, lexeme="/", literal="/"),
        EqlexToken(type=TokenType.NUMBER, lexeme="-10", literal="-10"), 
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_multichar_id(): 
    idscanner = Scanner("5 * AB")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"), 
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),  
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="AB", literal="AB"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_bern(): 
    idscanner = Scanner("0.25 + 0.5 * BERN(0.5)")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.BERN, lexeme="BERN", literal="BERN"),
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("),
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_norm(): 
    idscanner = Scanner("0.25 + 0.5 * NORM(0, 1)")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.NORMAL, lexeme="NORM", literal="NORM"),
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("),
        EqlexToken(type=TokenType.NUMBER, lexeme="0", literal="0"),
        EqlexToken(type=TokenType.COMMA, lexeme=",", literal=","),
        EqlexToken(type=TokenType.NUMBER, lexeme="1", literal="1"),
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_poisson(): 
    idscanner = Scanner("0.25 + 0.5 * POISSON(10 + A)")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.POISSON, lexeme="POISSON", literal="POISSON"),
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("),
        EqlexToken(type=TokenType.NUMBER, lexeme="10", literal="10"),
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_negbinom(): 
    idscanner = Scanner("0.25 + 0.5 * NEGBINOM(10, A)")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.NEGBIN, lexeme="NEGBINOM", literal="NEGBINOM"),
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("),
        EqlexToken(type=TokenType.NUMBER, lexeme="10", literal="10"),
        EqlexToken(type=TokenType.COMMA, lexeme=",", literal=","),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_weibull(): 
    idscanner = Scanner("0.25 + 0.5 * WEIBULL(10, A)")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.WEIBULL, lexeme="WEIBULL", literal="WEIBULL"),
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("),
        EqlexToken(type=TokenType.NUMBER, lexeme="10", literal="10"),
        EqlexToken(type=TokenType.COMMA, lexeme=",", literal=","),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"


    
def test_expon(): 
    idscanner = Scanner("0.25 + 0.5 * EXPON( A)")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.EXPON, lexeme="EXPON", literal="EXPON"),
        EqlexToken(type=TokenType.LEFT_PAREN, lexeme="(", literal="("),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.RIGHT_PAREN, lexeme=")", literal=")"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"


def test_ge(): 
    idscanner = Scanner("0.25 + 0.5 * A > 5")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.GT, lexeme=">", literal=">"),
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_gte(): 
    idscanner = Scanner("0.25 + 0.5 * A >= 5")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.GTE, lexeme=">=", literal=">="),
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"

def test_lt(): 
    idscanner = Scanner("0.25 + 0.5 * A < 5")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.LT, lexeme="<", literal="<"),
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"


def test_lte(): 
    idscanner = Scanner("0.25 + 0.5 * A <= 5")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="A", literal="A"),
        EqlexToken(type=TokenType.LTE, lexeme="<=", literal="<="),
        EqlexToken(type=TokenType.NUMBER, lexeme="5", literal="5"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"



    ...
# Edge cases: 
# 1. Invalid organization? 
# 2. Identifier names that overlap partially with distributions
def test_bernie(): 
    idscanner = Scanner("0.25 + 0.5 * BERNIE")

    expected_tokens: list[EqlexToken] = [
        EqlexToken(type=TokenType.NUMBER, lexeme="0.25", literal="0.25"), 
        EqlexToken(type=TokenType.PLUS, lexeme="+", literal="+"),  
        EqlexToken(type=TokenType.NUMBER, lexeme="0.5", literal="0.5"),
        EqlexToken(type=TokenType.STAR, lexeme="*", literal="*"),
        EqlexToken(type=TokenType.IDENTIFIER, lexeme="BERNIE", literal="BERNIE"),
        EqlexToken(type=TokenType.EOF, lexeme="", literal="")
    ]

    assert len(idscanner.token_list) == len(expected_tokens)
    for i, token in enumerate(idscanner.token_list): 
        assert token == expected_tokens[i], f"Expected {expected_tokens[i]}, received {token}"
        
# 3. Identifier names that overlap partially with functions
# 4. Identifier names with underscores? 