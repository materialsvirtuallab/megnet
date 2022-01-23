import unittest
import warnings
import multiprocessing
from monty.dev import deprecated, requires, get_ncpus, install_excepthook


class A:
    @property
    def repl_prop(self):
        pass

    @deprecated(repl_prop)  # type: ignore
    @property
    def prop(self):
        pass


class DecoratorTest(unittest.TestCase):
    def test_deprecated(self):
        def func_a():
            pass

        @deprecated(func_a, "hello")
        def func_b():
            pass

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            func_b()
            # Verify some things
            self.assertTrue(issubclass(w[0].category, FutureWarning))
            self.assertIn("hello", str(w[0].message))

    def test_deprecated_property(self):
        class a:
            def __init__(self):
                pass

            @property
            def property_a(self):
                pass

            @property  # type: ignore
            @deprecated(property_a)
            def property_b(self):
                return "b"

            @deprecated(property_a)
            def func_a(self):
                return "a"

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.assertEqual(a().property_b, "b")
            # Verify some things
            self.assertTrue(issubclass(w[-1].category, FutureWarning))

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.assertEqual(a().func_a(), "a")
            # Verify some things
            self.assertTrue(issubclass(w[-1].category, FutureWarning))

    def test_deprecated_classmethod(self):
        class A:
            def __init__(self):
                pass

            @classmethod
            def classmethod_a(self):
                pass

            @classmethod
            @deprecated(classmethod_a)
            def classmethod_b(self):
                return "b"

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.assertEqual(A().classmethod_b(), "b")
            # Verify some things
            self.assertTrue(issubclass(w[-1].category, FutureWarning))

        class A:
            def __init__(self):
                pass

            @classmethod
            def classmethod_a(self):
                pass

            @classmethod
            @deprecated(classmethod_a, category=DeprecationWarning)
            def classmethod_b(self):
                return "b"

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.assertEqual(A().classmethod_b(), "b")
            # Verify some things
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_requires(self):

        try:
            import fictitious_mod  # type: ignore
        except ImportError:
            fictitious_mod = None  # type: ignore

        @requires(fictitious_mod is not None, "fictitious_mod is not present.")
        def use_fictitious_mod():
            print("success")

        self.assertRaises(RuntimeError, use_fictitious_mod)

        @requires(unittest is not None, "scipy is not present.")
        def use_unittest():
            return "success"

        self.assertEqual(use_unittest(), "success")

    def test_get_ncpus(self):
        self.assertEqual(get_ncpus(), multiprocessing.cpu_count())

    def test_install_except_hook(self):
        install_excepthook()


if __name__ == "__main__":
    unittest.main()
