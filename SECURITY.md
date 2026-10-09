# Security

laya-ort runs local ONNX inference. It does not open a network service in this release.

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/Fruank4/laya-ort/security/advisories/new) for a problem that could leak data, load an unexpected file, or run attacker-controlled input unsafely. Do not open a public issue for an unfixed vulnerability.

Include the version or commit, the operating system, and a minimal way to reproduce the problem. A report that describes the impact is enough. A public exploit is not required.

## Out of scope

The official Laya weights are published by Convai Innovations. A problem in those weights or in upstream training belongs in [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). Model confidence is not a guarantee that an answer is correct.

## Supported versions

Security fixes go to the `main` branch of this repository.
