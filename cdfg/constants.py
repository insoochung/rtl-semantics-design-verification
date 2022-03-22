class Tag:
  MODULE = "kModuleDeclaration"
  MODULE_HEADER = "kModuleHeader"
  ALWAYS = "kAlwaysStatement"
  ALWAYS_CONTENT = "kProceduralTimingControlStatement"
  ALWAYS_CONDITION = "kEventControl"
  TERNARY_EXPRESSION = "kConditionExpression"
  IF_ELSE_STATEMENT = "kConditionalStatement"
  IF_CLAUSE = "kIfClause"
  IF_HEADER = "kIfHeader"
  IF_BODY = "kIfBody"
  ELSE_CLAUSE = "kElseClause"
  ELSE_BODY = "kElseBody"
  CASE_STATEMENT = "kCaseStatement"
  FOR_LOOP_STATEMENT = "kForLoopStatement"
  FOR_LOOP_CONDITION = "kLoopHeader"
  SEQ_BLOCK = "kSeqBlock"
  BLOCK_ITEM_LIST = "kBlockItemStatementList"
  STATEMENT = "kStatement"

  MACRO_GENERIC_ITEM = "kMacroGenericItem"

  NULL_STATEMENT = "kNullStatement"
  PROC_TIME_STATEMENT = "kProceduralTimingControlStatement"
  CASE_ITEM_LIST = "kCaseItemList"
  EXPRESSION_LIST = "kExpressionList"
  EXPRESSION = "kExpression"
  UNARY_EXPRESSION = "kUnaryExpression"
  UNARY_PREFIX_EXPRESSION = "kUnaryPrefixExpression"
  BINARY_EXPRESSION = "kBinaryExpression"
  CONCATENATE_EXPRESSION = "kConcatenationExpression"
  NUMBER = "kNumber"
  REFERENCE = "kReferenceCallBase"
  SYMBOL_IDENTIFIER = "SymbolIdentifier"
  LVALUE = "kLPValue"
  PARENTHESIS_GROUP = "kParenGroup"

  # Assignments
  ASSIGNMENT = "kNetVariableAssignment"
  ASSIGNMENT_MODIFY = "kAssignModifyStatement"
  NON_BLOCKING_ASSIGNMENT = "kNonblockingAssignmentStatement"

  # Keywords
  BEGIN = "kBegin"
  END = "kEnd"
  DEFAULT = "default"
  UNIQUE = "unique"
  CASE = "case"
  CASEZ = "casez"
  CASEX = "casex"
  ENDCASE = "endcase"
  COLON = ":"
  SEMICOLON = ";"
  ASSIGN = "="
  ASSIGN_NONBLOCK = "<="

  # Preprocessor related
  PREPROC_BALANCED = "kPreprocessorBalancedStatements"

  # Categories
  BRANCH_STATEMENTS = [IF_ELSE_STATEMENT, CASE_STATEMENT, TERNARY_EXPRESSION]
  CONDITION_STATEMENTS = [IF_HEADER, CASE_STATEMENT]
  BLOCK_STATEMENTS = BRANCH_STATEMENTS + [SEQ_BLOCK, FOR_LOOP_STATEMENT]
  TERMINAL_STATEMENTS = [ASSIGNMENT, ASSIGNMENT_MODIFY, PROC_TIME_STATEMENT,
                         NON_BLOCKING_ASSIGNMENT, STATEMENT, NULL_STATEMENT]
  ASSIGNMENTS = [ASSIGNMENT, ASSIGNMENT_MODIFY, NON_BLOCKING_ASSIGNMENT]
  ASSIGN_OPERATORS = [ASSIGN, ASSIGN_NONBLOCK]
  EXPRESSIONS = [EXPRESSION, UNARY_EXPRESSION, BINARY_EXPRESSION, REFERENCE,
                 CONCATENATE_EXPRESSION, NUMBER, PARENTHESIS_GROUP,
                 UNARY_PREFIX_EXPRESSION, TERNARY_EXPRESSION]
  IGNORE = [PREPROC_BALANCED]


class Condition:
  TRUE = "1"
  FALSE = "0"
  DEFAULT = "default"
  DATA = "data"
