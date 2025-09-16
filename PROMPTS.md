# Prompts

## Summary of `text_content.json`
```markdown
You are given a JSON document that contains elements extracted from a web page. Each element has metadata flags: isNav, isAction, isParagraph, isTitle, and possibly others. 
Your ultimate task is to write a concise summary of the meaningful content of the page. Follow these rules strictly:
1. Only include elements that represent actual content. These are:
   * Elements where isParagraph is true
   * Elements where isTitle is true
   * Elements where all meta flags are false
2. Exclude elements that are navigational or action-oriented. These are:
   * Elements where isNav is true
   * Elements where isAction is true
3. Do not generate or suggest actions.
4. Do not write code.
5. Do not use any external tools.
6. The output must be a plain-language summary that captures the main ideas and essence of the page for the user.

Summary:
```