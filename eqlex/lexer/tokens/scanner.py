from typing import List, Generator, Tuple
from eqlex.lexer.tokens.token import EqlexToken
from eqlex.lexer.tokens.token_types import TokenType
from dataclasses import dataclass, field

@dataclass
class Scanner():
    source: str
    token_list: List[EqlexToken] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.token_list = [tok for tok in self._generate_tokens() if tok is not None]

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
            return EqlexToken(type=TokenType.EOF, lexeme="", literal=""), idx, False

        
        current_char = self.source[idx]

        match current_char: 
            case " ": 
                # Ignore whitespace ...
                return None, idx + 1, True
            case "(" | ")" | "," | "." | "+" | "*" | "/" | "!" | "|" | "&":
                return EqlexToken(type=TokenType(current_char), lexeme=str(current_char), literal=str(current_char)), idx + 1, True

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
                    return EqlexToken(type=TokenType.NUMBER, lexeme=self.source[(idx):(end_idx-1)], literal=self.source[(idx):(end_idx-1)]), end_idx, True
                # Otherwise, should be subtraction
                return EqlexToken(type=TokenType.MINUS, lexeme=str(current_char), literal=str(current_char)), idx + 1, True
            
            case "=":
                if idx+1 < len(self.source) and self.source[idx+1] == "!": 
                    return EqlexToken(type=current_char, lexeme=str(current_char), literal=str(current_char)), idx + 1, True
               
                
            case ">":
                if idx+1 < len(self.source) and self.source[idx+1] == "=": 
                    return EqlexToken(type=TokenType.GTE, lexeme=self.source[idx:idx+2], literal=self.source[idx:idx+2]), idx+2, True
                return EqlexToken(type=TokenType.GT, lexeme=str(object=current_char), literal=str(object=current_char)), idx + 1, True
            case "<":
                if idx+1 < len(self.source) and self.source[idx+1] == "=": 
                    return EqlexToken(type=TokenType.LTE, lexeme=self.source[idx:idx+2], literal=self.source[idx:idx+2]), idx+2, True
                return EqlexToken(type=TokenType.LT, lexeme=str(object=current_char), literal=str(object=current_char)), idx + 1, True

            case _: 
                # Handle matching distributions
                if current_char.isalpha(): 

                    distribution_list = ["BERN", "NORM", "POISSON", "NEGBINOM", "WEIBULL", "EXPON"]
                    for distribution in distribution_list: 
                        if idx+len(distribution)+1 < len(self.source) and self.source[idx:(idx+len(distribution)+1)] == f"{distribution}(": 
                            return EqlexToken(TokenType(distribution), lexeme=distribution, literal=distribution), idx + len(distribution), True
                            
                    function_list = ["FLOOR", "CEIL", "LOG", "LOGIT", "EXP", "EXPIT", "SQRT", "POW"]
                    for func in function_list: 
                        if idx+len(func) < len(self.source) and self.source[idx:(idx+len(func)+1)] == f"{func}(": 
                            breakpoint()
                            return EqlexToken(TokenType(func), lexeme=func, literal=func), idx + len(func), True

                    end_idx = idx+1
                    while end_idx <= len(self.source) and self.source[(idx):(end_idx)].isalnum(): 
                        end_idx += 1
                        
                    return EqlexToken(type=TokenType.IDENTIFIER, lexeme=self.source[(idx):(end_idx-1)], literal=self.source[(idx):(end_idx-1)]), end_idx-1, True
                    
                # Handle literal numbers
                elif current_char.isdecimal():

                    end_idx = idx+1
                    while end_idx <= len(self.source) and self.source[(idx):(end_idx)].replace(".", "", count=1).isdecimal(): 
                        end_idx += 1
                    return EqlexToken(type=TokenType.NUMBER, lexeme=self.source[(idx):(end_idx-1)], literal=self.source[(idx):(end_idx-1)]), end_idx-1, True

                else: 
                    print(f"{current_char=}")
                    breakpoint()

        
        raise Exception(f"[Scanner::_get_next_token] Unmatched character: {current_char} with source {self.source} and idx {idx}")

