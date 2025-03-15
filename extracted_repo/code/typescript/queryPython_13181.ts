export const queryPython = `
(comment) @comment

(expression_statement
  (string) @comment) @docstring

(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
      (identifier) @name.reference.call
      (attribute
        attribute: (identifier) @name.reference.call)
  ]) @reference.call

(assignment
  left: (identifier) @name.definition.type_alias) @definition.type_alias
`;
