from typing import Dict, List, Generator, Tuple
from eqlex.lexer import EqlexToken, TokenType
from dataclasses import dataclass, field
from eqlex.lexer.shunting_yard import ReversePolishNotationList

PARENTHESIS_PRECEDENCE = 5
FUNC_PRECEDENCE = 4
MULT_PRECEDENCE = 3
ADDITION_PRECEDENCE = 2
LOGICAL_PRECEDENCE = 1 
LITERAL_PRECEDENCE = 0

@dataclass
class Scanner():
    source: str
    token_list: List[EqlexToken] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.token_list = [tok for tok in self._generate_tokens() if tok is not None]
        self.reverse_polish_notation = ReversePolishNotationList(input_token_list=self.token_list)

    def _generate_tokens(self) -> Generator[EqlexToken | None, None, None]:
        current: int = 0

        source_str: str = self.source
        hasNext: bool = len(source_str) > 0

        while hasNext:
            nextToken, current, hasNext = self._get_next_token(idx=current)
            yield nextToken


    def _get_next_token(self, idx: int) -> Tuple[EqlexToken | None, int, bool]:
        """
        Takes an input source and index, and returns the next token, next index, remaining source, and whether there are more items.
        """

        if idx >= len(self.source): 
            return EqlexToken(type=TokenType.EOF, lexeme=""), idx, False

        
        current_char = self.source[idx]

        match current_char: 
            case " ": 
                # Ignore whitespace ...
                return None, idx + 1, True
            case "(" | ")" | "." | ",":
                return EqlexToken(type=TokenType(current_char), lexeme=str(current_char), precedence=PARENTHESIS_PRECEDENCE), idx + 1, True
            case "|" | "&":
                return EqlexToken(type=TokenType(current_char), lexeme=current_char, n_args=1, precedence=LOGICAL_PRECEDENCE), idx + 1, True

            case "+": 
                return EqlexToken(type=TokenType(current_char), lexeme=current_char, n_args=2, precedence=ADDITION_PRECEDENCE), idx + 1, True

            case "*" | "/": 
                return EqlexToken(type=TokenType(current_char), lexeme=current_char, n_args=2, precedence=MULT_PRECEDENCE), idx + 1, True

            case "-":
                # Negative versus subtract
                if idx+1 < len(self.source) and self.source[idx+1].isdecimal(): 
                    # Parse as negative number
                    # Add 2 because need to offset 1 place for first negative number, and then an additional space to continue looking.
                    end_idx = idx+2
                    # Only accepts 1 decimal, which is correct.
                    while end_idx <= len(self.source) and self.source[(idx+1):(end_idx)].replace(".", "", count=1).isdecimal(): 
                        end_idx += 1

                    # However, at the end, return both the negative and what is beyond.
                    return EqlexToken(type=TokenType.NUMBER, lexeme=self.source[(idx):(end_idx-1)], precedence=LITERAL_PRECEDENCE), end_idx, True
                # Otherwise, should be subtraction
                return EqlexToken(type=TokenType.MINUS, lexeme=str(current_char), n_args=2, precedence=ADDITION_PRECEDENCE), idx + 1, True
            
            case "!":
                if idx+1 < len(self.source) and self.source[idx+1] == "=": 
                    return EqlexToken(type=TokenType.NEQ, lexeme="!=", n_args=2, precedence=LOGICAL_PRECEDENCE), idx + 1, True
                return EqlexToken(type=TokenType.BANG, lexeme=current_char, n_args=1, precedence=LOGICAL_PRECEDENCE), idx + 1, True

            case "=": 
                if idx+1 < len(self.source) and self.source[idx+1] == "=": 
                    return EqlexToken(type=TokenType.EQ_EQ, lexeme="==", n_args=2, precedence=LOGICAL_PRECEDENCE), idx + 1, True

            case ">":
                if idx+1 < len(self.source) and self.source[idx+1] == "=": 
                    return EqlexToken(type=TokenType.GTE, lexeme=self.source[idx:idx+2], n_args=2, precedence=LOGICAL_PRECEDENCE), idx+2, True
                return EqlexToken(type=TokenType.GT, lexeme=str(object=current_char), n_args=2, precedence=LOGICAL_PRECEDENCE), idx + 1, True

            case "<":
                if idx+1 < len(self.source) and self.source[idx+1] == "=": 
                    return EqlexToken(type=TokenType.LTE, lexeme=self.source[idx:idx+2], n_args=2, precedence=LOGICAL_PRECEDENCE), idx+2, True
                return EqlexToken(type=TokenType.LT, lexeme=str(object=current_char), n_args=2, precedence=LOGICAL_PRECEDENCE), idx + 1, True

            case _: 
                # Handle matching distributions
                if current_char.isalpha(): 
                    
                    distribution_dict: Dict[str, int] = {
                       "BERN": 1, 
                       "NORM": 2, 
                       "POISSON": 1, 
                       "NEGBINOM": 2, 
                       "WEIBULL": 2, 
                       "EXPON": 1
                    }
                    for distribution, nargs in distribution_dict.items(): 
                        if idx+len(distribution)+1 < len(self.source) and self.source[idx:(idx+len(distribution)+1)] == f"{distribution}(": 
                            return EqlexToken(TokenType(distribution), lexeme=distribution, n_args=nargs, precedence=FUNC_PRECEDENCE), idx + len(distribution), True

                    function_dict = {
                        "FLOOR": 1,
                        "CEIL": 1,
                        "LOG": 1, 
                        "LOGIT": 1, 
                        "EXP": 1, 
                        "EXPIT": 1, 
                        "SQRT": 1, 
                        "POW": 2
                    } 
                    for func, nargs in function_dict.items(): 
                        if idx+len(func) < len(self.source) and self.source[idx:(idx+len(func)+1)] == f"{func}(": 
                            return EqlexToken(TokenType(func), lexeme=func, n_args=nargs, precedence=FUNC_PRECEDENCE), idx + len(func), True

                    end_idx = idx+1
                    while end_idx <= len(self.source) and self.source[(idx):(end_idx)].isalnum(): 
                        end_idx += 1
                        
                    return EqlexToken(type=TokenType.IDENTIFIER, lexeme=self.source[(idx):(end_idx-1)], precedence=LITERAL_PRECEDENCE), end_idx-1, True
                    
                # Handle literal numbers
                elif current_char.isdecimal():

                    end_idx = idx+1
                    while end_idx <= len(self.source) and self.source[(idx):(end_idx)].replace(".", "", count=1).isdecimal(): 
                        end_idx += 1
                    return EqlexToken(type=TokenType.NUMBER, lexeme=self.source[(idx):(end_idx-1)], precedence=LITERAL_PRECEDENCE), end_idx-1, True

                else: 
                    print(f"{current_char=}")
                    breakpoint()

        
        raise Exception(f"[Scanner::_get_next_token] Unmatched character: {current_char} with source {self.source} and idx {idx}")
