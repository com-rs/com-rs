# Security policy

## Supported versions

com-rs is in alpha: only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Report the problem privately through the GitHub
security advisories of the repository: on
<https://github.com/com-rs/com-rs/security/advisories/new>, or
**Security → Report a vulnerability** in the repository. Include the version, the
platform, and the input (a specification, a recording or a script) that reproduces it.

You will receive an answer within two weeks. Once a fix is released, the advisory is
published with credit to the reporter, unless they prefer otherwise.

com-rs reads files given by its user (specifications in YAML, JSON or TOML; SigMF, raw
IQ and NPY recordings): a crash, a hang or an unbounded allocation on a malformed file
is worth reporting.
