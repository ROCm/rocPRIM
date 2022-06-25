from sphinx.writers.html5 import HTML5Translator
from docutils.nodes import Element


class ROCPrimHTMLTranslator(HTML5Translator):
    """
    This translator is a hack that ameliorates the way the function signatures
    are rendered.

    - Render the function parameters as a HTML list.
    - Divide template function signatures into two HTML spans, one with the
      template part and one with the function signature.

    It is coupled with the CSS `static/cpp_sig.css`.
    """

    def visit_desc_signature_line(self, node: Element) -> None:
        super().visit_desc_signature_line(node)

        if node.sphinx_line_type == "templateParams":
            self.body.append('<span class="template-params">')
        elif  node.sphinx_line_type == "declarator":
            self.body.append('<span class="declarator">')

    def depart_desc_signature_line(self, node: Element) -> None:
        super().depart_desc_signature_line(node)

        if node.sphinx_line_type in ("templateParams", "declarator"):
            self.body.append('</span>')

    def visit_desc_parameterlist(self, node: Element) -> None:
        super().visit_desc_parameterlist(node)

        if node.children:
            self.body.append('<ul class="desc-parameterlist">')

    def depart_desc_parameterlist(self, node: Element) -> None:
        if node.children:
            self.body.append('</ul>')

        super().depart_desc_parameterlist(node)

    def visit_desc_parameter(self, node: Element) -> None:
        self.body.append('<li class="desc-parameter">')

        super().visit_desc_parameter(node)

    def depart_desc_parameter(self, node: Element) -> None:
        super().depart_desc_parameter(node)

        self.body.append("</li>")
