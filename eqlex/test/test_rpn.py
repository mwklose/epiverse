from eqlex.lexer.shunting_yard import ReversePolishNotationList
from eqlex.lexer import EqlexToken, TokenType
from eqlex.lexer.scanner import ADDITION_PRECEDENCE, FUNC_PRECEDENCE, MULT_PRECEDENCE, PARENTHESIS_PRECEDENCE

def test_one_two_plus(): 
    rpn = ReversePolishNotationList([
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "2")
    ])

    expected_rpn = [
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE)
    ]

    for i, rpn_token in enumerate(rpn.rpn): 
        assert rpn_token == expected_rpn[i], f"Expected {expected_rpn[i]}, received {rpn_token}"

def test_add_subtract(): 
    rpn = ReversePolishNotationList([
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.MINUS, "-", 2, ADDITION_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "4")

    ])

    expected_rpn = [
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE),
        EqlexToken(TokenType.NUMBER, "4"),
        EqlexToken(TokenType.MINUS, "-", 2, ADDITION_PRECEDENCE)
    ]

    for i, rpn_token in enumerate(rpn.rpn): 
        assert rpn_token == expected_rpn[i], f"Expected {expected_rpn[i]}, received {rpn_token}"

def test_add_mult(): 
    rpn = ReversePolishNotationList([
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.STAR, "*", 2, MULT_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "4")

    ])

    expected_rpn = [
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.NUMBER, "4"),
        EqlexToken(TokenType.STAR, "*", 2, MULT_PRECEDENCE),
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE)

    ]

    for i, rpn_token in enumerate(rpn.rpn): 
        assert rpn_token == expected_rpn[i], f"Expected {expected_rpn[i]}, received {rpn_token}"

def test_add_mult_paren(): 
    rpn = ReversePolishNotationList([
        EqlexToken(TokenType.LEFT_PAREN, "(", precedence=PARENTHESIS_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.RIGHT_PAREN, ")", precedence=PARENTHESIS_PRECEDENCE),
        EqlexToken(TokenType.STAR, "*", 2, MULT_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "4")

    ])

    expected_rpn = [
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE),
        EqlexToken(TokenType.NUMBER, "4"),
        EqlexToken(TokenType.STAR, "*", 2, MULT_PRECEDENCE)
    ]

    for i, rpn_token in enumerate(rpn.rpn): 
        assert rpn_token == expected_rpn[i], f"Expected {expected_rpn[i]}, received {rpn_token}"

def test_bern_func(): 
    rpn = ReversePolishNotationList([
        EqlexToken(TokenType.LEFT_PAREN, "(", precedence=PARENTHESIS_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.RIGHT_PAREN, ")", precedence=PARENTHESIS_PRECEDENCE),
        EqlexToken(TokenType.STAR, "*", 2, MULT_PRECEDENCE), 
        EqlexToken(TokenType.BERN, "BERN", n_args=1, precedence=FUNC_PRECEDENCE),
        EqlexToken(TokenType.LEFT_PAREN, "(", precedence=PARENTHESIS_PRECEDENCE), 
        EqlexToken(TokenType.NUMBER, "0.5"),
        EqlexToken(TokenType.RIGHT_PAREN, ")", precedence=PARENTHESIS_PRECEDENCE)
    ])

    expected_rpn = [
        EqlexToken(TokenType.NUMBER, "1"), 
        EqlexToken(TokenType.NUMBER, "2"),
        EqlexToken(TokenType.PLUS, "+", 2, ADDITION_PRECEDENCE),
        EqlexToken(TokenType.NUMBER, "0.5"),
        EqlexToken(TokenType.BERN, "BERN", n_args=1, precedence=FUNC_PRECEDENCE),
        EqlexToken(TokenType.STAR, "*", 2, MULT_PRECEDENCE)
    ]

    for i, rpn_token in enumerate(rpn.rpn): 
        assert rpn_token == expected_rpn[i], f"Expected {expected_rpn[i]}, received {rpn_token}"