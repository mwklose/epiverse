from dataclasses import dataclass
from eqlex.lexer.token_types import TokenType

@dataclass
class EqlexToken(): 
    type: TokenType
    lexeme: str
    n_args: int = 0
    precedence: int = 0

    def __str__(self) -> str:
        return f"[{self.type}] {self.lexeme}"

    