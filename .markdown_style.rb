all
# Match 2-space list indentation standard
rule 'ul-indent', :indent => 2
# Allow ordered list numbers
rule 'ol-prefix', :style => :ordered
# Enforce 80-character line length on narrative prose; exempt code blocks and tables
rule 'line-length', :line_length => 80, :ignore_code_blocks => true, :tables => false
# Allow inline HTML
exclude_rule 'no-inline-html'
# Allow indented list continuations inside blockquotes / GitHub alerts
exclude_rule 'no-multiple-space-blockquote'
# Disable broken rules removed from latest release
# https://github.com/markdownlint/markdownlint/issues/472
exclude_rule 'MD055'
exclude_rule 'MD056'
exclude_rule 'MD057'
