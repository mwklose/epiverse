from typing import List
from eqlex.lexer import EqlexToken, TokenType
from dataclasses import dataclass

@dataclass
class ReversePolishNotationList(): 
    input_token_list: List[EqlexToken]

    def __post_init__(self): 
        self.rpn: List[EqlexToken] = self.shunting_yard()

    def shunting_yard(self) -> List[EqlexToken]:
        values_stack: List[EqlexToken] = []
        operators_stack: List[EqlexToken] = []

        for token in self.input_token_list: 
            if token.type in [TokenType.IDENTIFIER, TokenType.NUMBER]:
                values_stack.append(token)
            elif token.type in [TokenType.LEFT_PAREN]:
                operators_stack.append(token)
            elif token.type in [TokenType.RIGHT_PAREN]:
                empty_operator_stack = len(operators_stack) == 0
                found_left_parethesis = False

                while not empty_operator_stack: 
                    if len(operators_stack) == 0: 
                        raise Exception(f"[ReversePolishNotation] Could not find match for right parenthesis; input list {self.input_token_list}")
                    
                    if operators_stack[-1].type == TokenType.LEFT_PAREN: 
                        found_left_parethesis = True
                        operators_stack.pop()
                        break

                    values_stack.append(operators_stack.pop())

                if not found_left_parethesis: 
                    breakpoint()
            elif token.type in [TokenType.COMMA]: 
                continue
            else: 
                # Do shunting yard
                if len(operators_stack) == 0: 
                    operators_stack.append(token)
                    continue

                top_is_left_paren = operators_stack[-1].type == TokenType.LEFT_PAREN 
                if top_is_left_paren: 
                    operators_stack.append(token)
                elif operators_stack[-1].precedence < token.precedence: 
                    operators_stack.append(token)
                else: 
                    while len(operators_stack) > 0 and operators_stack[-1].precedence >= token.precedence: 
                        values_stack.append(operators_stack.pop())
                    operators_stack.append(token)
                
        while len(operators_stack) > 0: 
            values_stack.append(operators_stack.pop())

        return values_stack