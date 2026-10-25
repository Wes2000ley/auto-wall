# Bundled dependencies

All runtime code is local. The manual editor does not read map pixels, run image
kernels, start detection workers, download models, or use a companion service.

| File | Source | Version | License |
| --- | --- | --- | --- |
| `vendor/polygon-clipping.js` | [Upstream UMD distribution](https://github.com/mfogel/polygon-clipping/blob/v0.15.7/dist/polygon-clipping.umd.js) | 0.15.7 | MIT, `vendor/polygon-clipping-LICENSE.md` |

Pinned SHA-256:

```text
4b1c62d5e76ba6d384f29b548105d8738a9dd54beccc62fca4bfd8235c68cc92  vendor/polygon-clipping.js
```

The original download hash was
`de2bdc061b61d537f7d12f6519f46e35113f0f246b6a06293a08b615454946cf`;
its mixed CRLF/LF comment lines were normalized to LF without changing logic.
Updating the dependency requires updating the hash and rerunning geometry tests.
The module itself follows the upstream Auto-Wall MIT license in `LICENSE`.

Polygon-clipping includes splaytree 3.1.2 (MIT), robust-predicates 3.0.2
(Unlicense/public-domain dedication), and Microsoft TypeScript helper code
(Apache-2.0). Their notices remain in the distribution. The full splaytree and
robust-predicates license texts are alongside it; `Apache-2.0.txt` preserves the
Apache license for the helper code. Splaytree's text comes from its published
3.1.2 Readme license section; robust-predicates' text comes from its v3.0.2 source.
