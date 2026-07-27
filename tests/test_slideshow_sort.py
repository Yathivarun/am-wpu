"""_natural_key regression coverage.

README changelog (2026-07-11): sort_mode "numeric" used to raise
`TypeError: '<' not supported between instances of 'int' and 'str'` the
moment a non-numerically-named image sat alongside numerically-named ones,
because the old key returned an int for one file and a str for another.
`_natural_key` fixed that by grouping numeric and non-numeric names into
disjoint tuple buckets so they're never compared cross-type.
"""

from wpu_client.services.slideshow.slideshow_service import _natural_key


def test_numeric_names_sort_numerically_not_lexically():
    paths = ["/a/10.png", "/a/2.png", "/a/1.png"]
    assert sorted(paths, key=_natural_key) == ["/a/1.png", "/a/2.png", "/a/10.png"]


def test_numeric_names_sort_before_non_numeric_names():
    paths = ["/a/banner.png", "/a/3.png"]
    assert sorted(paths, key=_natural_key) == ["/a/3.png", "/a/banner.png"]


def test_mixed_directory_does_not_raise():
    # This exact mix used to crash with the int/str TypeError.
    paths = ["/a/2.png", "/a/logo.png", "/a/1.png", "/a/cover.jpg"]
    result = sorted(paths, key=_natural_key)
    assert result == ["/a/1.png", "/a/2.png", "/a/cover.jpg", "/a/logo.png"]


def test_non_numeric_names_sort_case_insensitively():
    paths = ["/a/Banner.png", "/a/apple.png"]
    assert sorted(paths, key=_natural_key) == ["/a/apple.png", "/a/Banner.png"]
