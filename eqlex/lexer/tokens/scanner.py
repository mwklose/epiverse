from typing import List
from eqlex.lexer.tokens.token import Token
from dataclasses import dataclass, field

@dataclass
class Scanner(): 
    source: str
    token_list: List[Token] = field(default_factory=list)

    def __post_init__(self) -> None: 

        self.token_list = [tok for tok in self._generate_tokens()]
        
    def _generate_tokens(self): 
        
        
        
        ...
        