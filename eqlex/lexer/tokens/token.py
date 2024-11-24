from dataclasses import dataclass
from eqlex.lexer.tokens.token_types import TokenType

@dataclass
class EqlexToken(): 
    type: TokenType
    lexeme: str
    literal: str

    def __str__(self) -> str:
        return f"[{self.type}] {self.lexeme}"

    