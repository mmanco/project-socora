/**
 * @param {string} xpath
 * @param {Node} contextNode
 */
function evaluateXPath(xpath, contextNode) {
    const evaluator = new XPathEvaluator();
    const result = evaluator.evaluate(
        xpath,
        contextNode,
        null,
        XPathResult.ANY_TYPE,
        null
    );
    return result;
}