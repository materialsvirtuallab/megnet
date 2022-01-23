# Python-RSA changelog

## Version 4.8 - in development

- Switch to [Poetry](https://python-poetry.org/) for dependency and release management.
- Compatibility with Python 3.10.
- Chain exceptions using `raise new_exception from old_exception`
  ([#157](https://github.com/sybrenstuvel/python-rsa/pull/157))
- Added marker file for PEP 561. This will allow type checking tools in dependent projects
  to use type annotations from Python-RSA
  ([#136](https://github.com/sybrenstuvel/python-rsa/pull/136)).
- Use the Chinese Remainder Theorem when decrypting with a private key. This
  makes decryption 2-4x faster
  ([#163](https://github.com/sybrenstuvel/python-rsa/pull/163)).

## Version 4.7.2 - released 2021-02-24

- Fix picking/unpickling issue introduced in 4.7
  ([#173](https://github.com/sybrenstuvel/python-rsa/issues/173))

## Version 4.7.1 - released 2021-02-15

- Fix threading issue introduced in 4.7
  ([#173](https://github.com/sybrenstuvel/python-rsa/issues/173))

## Version 4.7 - released 2021-01-10

- Fix [#165](https://github.com/sybrenstuvel/python-rsa/issues/165):
  CVE-2020-25658 - Bleichenbacher-style timing oracle in PKCS#1 v1.5 decryption
  code
- Add padding length check as described by PKCS#1 v1.5 (Fixes
  [#164](https://github.com/sybrenstuvel/python-rsa/issues/164))
- Reuse of blinding factors to speed up blinding operations.
  Fixes [#162](https://github.com/sybrenstuvel/python-rsa/issues/162).
- Declare & test support for Python 3.9


## Version 4.4 & 4.6 - released 2020-06-12

Version 4.4 and 4.6 are almost a re-tagged release of version 4.2. It requires
Python 3.5+. To avoid older Python installations from trying to upgrade to RSA
4.4, this is now made explicit in the `python_requires` argument in `setup.py`.
There was a mistake releasing 4.4 as "3.5+ only", which made it necessary to
retag 4.4 as 4.6 as well.

No functional changes compared to version 4.2.


## Version 4.3 & 4.5 - released 2020-06-12

Version 4.3 and 4.5 are almost a re-tagged release of version 4.0. It is the
last to support Python 2.7. This is now made explicit in the `python_requires`
argument in `setup.py`. Python 3.4 is not supported by this release. There was a
mistake releasing 4.4 as "3.5+ only", which made it necessary to retag 4.3 as
4.5 as well.

Two security fixes have also been backported, so 4.3 = 4.0 + these two fixes.

- Choose blinding factor relatively prime to N. Thanks Christian Heimes for pointing this out.
- Reject cyphertexts (when decrypting) and signatures (when verifying) that have
  been modified by prepending zero bytes. This resolves CVE-2020-13757. Thanks
  Carnil for pointing this out.


## Version 4.2 - released 2020-06-10

- Rolled back the switch to Poetry, and reverted back to using Pipenv + setup.py
  for dependency management. There apparently is an issue no-binary installs of
  packages build with Poetry. This fixes
  [#148](https://github.com/sybrenstuvel/python-rsa/issues/148)
- Limited SHA3 support to those Python versions (3.6+) that support it natively.
  The third-party library that adds support for this to Python 3.5 is a binary
  package, and thus breaks the pure-Python nature of Python-RSA.
  This should fix [#147](https://github.com/sybrenstuvel/python-rsa/issues/147).


## Version 4.1 - released 2020-06-10

- Added support for Python 3.8.
- Dropped support for Python 2 and 3.4.
- Added type annotations to the source code. This will make Python-RSA easier to use in
  your IDE, and allows better type checking.
- Added static type checking via [MyPy](http://mypy-lang.org/).
- Fix [#129](https://github.com/sybrenstuvel/python-rsa/issues/129) Installing from source
  gives UnicodeDecodeError.
- Switched to using [Poetry](https://poetry.eustace.io/) for package
  management.
- Added support for SHA3 hashing: SHA3-256, SHA3-384, SHA3-512. This
  is natively supported by Python 3.6+ and supported via a third-party
  library on Python 3.5.
- Choose blinding factor relatively prime to N. Thanks Christian Heimes for pointing this out.
- Reject cyphertexts (when decrypting) and signatures (when verifying) that have
  been modified by prepending zero bytes. This resolves CVE-2020-13757. Thanks
  Adelapie for pointing this out.


## Version 4.0 - released 2018-09-16

- Removed deprecated modules:
    - rsa.varblock
    - rsa.bigfile
    - rsa._version133
    - rsa._version200
- Removed CLI commands that use the VARBLOCK/bigfile format.
- Ensured that PublicKey.save_pkcs1() and PrivateKey.save_pkcs1() always return bytes.
- Dropped support for Python 2.6 and 3.3.
- Dropped support for Psyco.
- Miller-Rabin iterations determined by bitsize of key.
  [#58](https://github.com/sybrenstuvel/python-rsa/pull/58)
- Added function `rsa.find_signature_hash()` to return the name of the hashing
  algorithm used to sign a message. `rsa.verify()` now also returns that name,
  instead of always returning `True`.
  [#78](https://github.com/sybrenstuvel/python-rsa/issues/13)
- Add support for SHA-224 for PKCS1 signatures.
  [#104](https://github.com/sybrenstuvel/python-rsa/pull/104)
- Transitioned from `requirements.txt` to Pipenv for package management.


## Version 3.4.2 - released 2016-03-29

- Fixed dates in CHANGELOG.txt


## Version 3.4.1 - released 2016-03-26

- Included tests/private.pem in MANIFEST.in
- Included README.md and CHANGELOG.txt in MANIFEST.in


## Version 3.4 - released 2016-03-17

- Moved development to GitHub: https://github.com/sybrenstuvel/python-rsa
- Solved side-channel vulnerability by implementing blinding, fixes #19
- Deprecated the VARBLOCK format and rsa.bigfile module due to security issues, see
    https://github.com/sybrenstuvel/python-rsa/issues/13
- Integration with Travis-CI [1], Coveralls [2] and Code Climate [3]
- Deprecated the old rsa._version133 and rsa._version200 submodules, they will be
  completely removed in version 4.0.
- Add an 'exponent' argument to key.newkeys()
- Switched from Solovay-Strassen to Miller-Rabin primality testing, to
  comply with NIST FIPS 186-4 [4] as probabilistic primality test
  (Appendix C, subsection C.3):
- Fixed bugs #12, #14, #27, #30, #49

[1] https://travis-ci.org/sybrenstuvel/python-rsa
[2] https://coveralls.io/github/sybrenstuvel/python-rsa
[3] https://codeclimate.com/github/sybrenstuvel/python-rsa
[4] http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf


## Version 3.3 - released 2016-01-13

- Thanks to Filippo Valsorda: Fix BB'06 attack in verify() by
  switching from parsing to comparison. See [1] for more information.
- Simplified Tox configuration and dropped Python 3.2 support. The
  coverage package uses a u'' prefix, which was reintroduced in 3.3
  for ease of porting.

[1] https://blog.filippo.io/bleichenbacher-06-signature-forgery-in-python-rsa/


## Version 3.2.3 - released 2015-11-05

- Added character encoding markers for Python 2.x


## Version 3.2.1 - released 2015-11-05

- Added per-file licenses
- Added support for wheel packages
- Made example code more consistent and up to date with Python 3.4


## Version 3.2 - released 2015-07-29

- Mentioned support for Python 3 in setup.py


## Version 3.1.4 - released 2014-02-22

- Fixed some bugs


## Version 3.1.3 - released 2014-02-02

- Dropped support for Python 2.5


## Version 3.1.2 - released 2013-09-15

- Added Python 3.3 to the test environment.
- Removed dependency on Distribute
- Added support for loading public keys from OpenSSL


## Version 3.1.1 - released 2012-06-18

- Fixed doctests for Python 2.7
- Removed obsolete unittest so all tests run fine on Python 3.2

## Version 3.1 - released 2012-06-17

- Big, big credits to Yesudeep Mangalapilly for all the changes listed
  below!
- Added ability to generate keys on multiple cores simultaneously.
- Massive speedup
- Partial Python 3.2 compatibility (core functionality works, but
  saving or loading keys doesn't, for that the pyasn1 package needs to
  be ported to Python 3 first)
- Lots of bug fixes



## Version 3.0.1 - released 2011-08-07

- Removed unused import of abc module


## Version 3.0 - released 2011-08-05

- Changed the meaning of the keysize to mean the size of ``n`` rather than
  the size of both ``p`` and ``q``. This is the common interpretation of
  RSA keysize. To get the old behaviour, double the keysize when generating a
  new key.
- Added a lot of doctests
- Added random-padded encryption and decryption using PKCS#1 version 1.5
- Added hash-based signatures and verification using PKCS#1v1.5
- Modeling private and public key as real objects rather than dicts.
- Support for saving and loading keys as PEM and DER files.
- Ability to extract a public key from a private key (PEM+DER)


## Version 2.0

- Security improvements by Barry Mead.
