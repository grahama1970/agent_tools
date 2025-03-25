from tree_sitter_language_pack import get_binding, get_language, get_parser

ada_binding = get_binding('ada')  # this is a pycapsule object pointing to the C binding
ada_lang = get_language('ada')  # this is an instance of tree_sitter.Language
ada_parser = get_parser('ada')  # this is an instance of tree_sitter.Parser



# def get_parser(language_name: SupportedLanguage) -> Parser:
#     """Get a parser for the given language name.

#     Args:
#         language_name: The name of the language.

#     Returns:
#         Parser: The parser for the language as a tree-sitter Parser instance.
#     """
#     return Parser(get_language(language_name=language_name))