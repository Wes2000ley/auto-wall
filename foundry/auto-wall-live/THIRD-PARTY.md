# Bundled dependencies

All runtime code is local to this module. No CDN, model download, telemetry,
Python service or GPU runtime is required. OpenCV is initialized in a dedicated
worker only when Detect map is used; the manual tools do not initialize it.

| File | Source | Version | License |
| --- | --- | --- | --- |
| `vendor/opencv.js` | [Official OpenCV prebuilt JavaScript/WASM](https://docs.opencv.org/4.x/opencv.js) | OpenCV 4.13.0; build timestamp 2025-12-31 | Apache-2.0, `vendor/OpenCV-LICENSE.txt` |
| `vendor/polygon-clipping.js` | [Upstream UMD distribution](https://github.com/mfogel/polygon-clipping/blob/v0.15.7/dist/polygon-clipping.umd.js) | 0.15.7 | MIT, `vendor/polygon-clipping-LICENSE.md`; bundled dependency notices remain in the source |

Pinned SHA-256:

```text
e03dbafdbea1bfc123478a811571191b3626a5dc13e8cc71251a871e4ded0213  vendor/opencv.js
4b1c62d5e76ba6d384f29b548105d8738a9dd54beccc62fca4bfd8235c68cc92  vendor/polygon-clipping.js
```

The OpenCV documentation URL follows the 4.x series, so the checked-in bytes and
hash pin this particular build. The downloaded source SHA-256 was
`63366510248adf3a7eddf3e793dd825404efb7df3749f4d6f8557c7fa4ca8aa0`;
the local copy only removes trailing whitespace in blank wrapper lines and the
extra blank line at EOF. The JavaScript logic and inline WASM bytes are unchanged.
The polygon-clipping download's SHA-256 was
`de2bdc061b61d537f7d12f6519f46e35113f0f246b6a06293a08b615454946cf`;
its mixed CRLF/LF comment lines were normalized to LF without changing logic.
Updating the dependency requires updating the
hash and rerunning the WASM tests. The OpenCV source retains its own third-party
notices. The module itself follows the upstream Auto-Wall MIT license in `LICENSE`.

Polygon-clipping's distribution includes splaytree 3.1.2 (MIT), robust-predicates
3.0.2 (Unlicense/public-domain dedication), and Microsoft TypeScript helper code
(Apache-2.0). The corresponding notices are retained in the distribution, with
the full splaytree and robust-predicates texts alongside it. The Apache-2.0 text
is included in `OpenCV-LICENSE.txt`. Splaytree's text comes from its published
3.1.2 Readme license section; robust-predicates' text comes from its v3.0.2 source.
