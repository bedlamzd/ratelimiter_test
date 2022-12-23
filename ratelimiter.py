import time
from dataclasses import InitVar, dataclass
from functools import cached_property
from typing import Final, TypeAlias

FractionalSeconds: TypeAlias = float
NanoSeconds: TypeAlias = int
_SECOND_AS_NS: Final[NanoSeconds] = 1_000_000_000


@dataclass(frozen=True)
class Rate:
    """
    How often something happens with period expressed in nanoseconds

    Normalized after creation, so that ``period_ns`` corresponds to
    ``amount`` of 1.

    >>> one_per_second = Rate()  # default is once per second
    >>> assert one_per_second.period_ns == _SECOND_AS_NS

    >>> two_per_second = Rate(2, _SECOND_AS_NS)
    >>> assert two_per_second.period_ns == (_SECOND_AS_NS // 2)

    >>> half_second_as_ns = _SECOND_AS_NS // 2
    >>> one_per_half_second = Rate(1, half_second_as_ns)
    >>> assert one_per_half_second.period_ns == half_second_as_ns
    >>> assert one_per_half_second == two_per_second
    """
    amount: InitVar[int] = 1
    period_ns: NanoSeconds = _SECOND_AS_NS

    def __post_init__(self, amount: int) -> None:
        if amount < 1:
            raise ValueError(f"`amount` should be integer >= 1, got {amount}")
        object.__setattr__(self, "period_ns", int(self.period_ns / amount))

    @classmethod
    def from_frequency(cls, frequency_per_sec: float) -> 'Rate':
        # `Self` would be better as a return type, but for 3.10 it still requires
        # `typing_extension` which is overkill for test task.
        # Also, not using `from __future__ import annotations` due to
        # bug with `dataclass.InitVar`
        # (most likely related to https://github.com/python/cpython/issues/88580)
        """
        Make instance given frequency value

        >>> rate = Rate()  # 1 rps
        >>> from_freq = Rate.from_frequency(1)  # 1 rps
        >>> assert rate == from_freq

        >>> rate = Rate(1, 2 * _SECOND_AS_NS)  # 0.5 rps
        >>> from_freq = Rate.from_frequency(0.5)  # 0.5 rps
        >>> assert rate == from_freq

        :param frequency_per_sec: How often something happens per second
        :return: Rate
        """
        period_sec = 1 / frequency_per_sec
        period_ns = int(period_sec * _SECOND_AS_NS)
        return cls(1, period_ns)

    @cached_property
    def period_sec(self) -> FractionalSeconds:
        """
        Refill preiod as a fractional second

        >>> import math
        >>> rr = Rate()
        >>> assert math.isclose(rr.period_sec, 1.0)

        >>> rr = Rate(2)
        >>> assert math.isclose(rr.period_sec, 0.5)

        >>> rr = Rate(3, 60 * _SECOND_AS_NS)
        >>> assert math.isclose(rr.period_sec, 20.0)

        :return: period in fractional seconds
        """
        return self.period_ns / _SECOND_AS_NS


_ONE_PER_SECOND: Final[Rate] = Rate()


class Bucket:
    """
    See `token bucket`_ algorithm for more.

    Uses ``time.monotonic_ns`` to be resilient to system clock changes.
    For example correction via NTP. See `PEP 418`_ for more.

    .. _token bucket: https://en.wikipedia.org/wiki/Token_bucket
    .. _PEP 418: https://peps.python.org/pep-0418/
    """
    _volume: Final[int]
    _refill_rate: Final[Rate]
    _current: int
    _last_refill_ns: NanoSeconds

    def __init__(self, volume: int = 1, refill_rate: Rate = _ONE_PER_SECOND) -> None:
        self._volume = volume
        self._current = volume
        self._refill_rate = refill_rate
        self._last_refill_ns = time.monotonic_ns()

    @property
    def is_empty(self) -> bool:
        """
        Return whether `Bucket` has any tokens left or not

        >>> b = Bucket(10)
        >>> b.is_empty
        False
        >>> b.remove(10)
        >>> b.is_empty
        True
        >>> b.remove(10)  # go into negatives
        >>> b.is_empty
        True
        >>>

        :return: bool
        """
        return self._current <= 0

    def remove(self, amount: int = 1, /) -> None:
        """
        Remove ``amount`` of tokens from the bucket.

        Can result in negative amount of tokens in the bucket.

        >>> volume = 5
        >>> b = Bucket(volume)
        >>> b.remove(-1)
        Traceback (most recent call last):
            ...
        ValueError: Can remove only positive amount, got -1!
        >>> b.remove()
        >>> b._current
        4
        >>> b.remove(3)
        >>> b._current
        1
        >>> b.remove(10)  # can go into negatives
        >>> b._current
        -9
        >>>

        :param amount: How many tokens to remove
        :return: None
        """

        if amount < 0:
            raise ValueError(f"Can remove only positive amount, got {amount}!")

        self._current -= amount

    def refill(self) -> None:
        """
        Refill the bucket according to its refill rate.

        >>> volume, refill_rate = 1, Rate(1, 2 * _SECOND_AS_NS)
        >>> b = Bucket(volume, refill_rate=refill_rate)

        If bucket already full, simply update refill time.
        >>> last_update = b._last_refill_ns
        >>> for _ in range(5):
        ...     b.refill()
        ...     assert b._current == volume
        ...     assert b._last_refill_ns != last_update
        ...     last_update = b._last_refill_ns

        Tries to not lose any tokens.
        >>> import math
        >>> volume, refill_rate = 2, Rate(1, 2 * _SECOND_AS_NS)
        >>> b = Bucket(volume, refill_rate)
        >>> last_update = b._last_refill_ns
        >>> b.remove(volume)  # remove all tokens
        >>> b.is_empty
        True
        >>> # wait more than refill period
        >>> time.sleep(1.5 * refill_rate.period_sec)
        >>> b.refill()  # should add only one token
        >>> b.is_empty, b._current
        (False, 1)
        >>> # last refill time should be behind current time for half a period
        >>> # or, equivalently, difference between updates should only be of 1 period
        >>> math.isclose(
        ...     last_update + refill_rate.period_ns, b._last_refill_ns, rel_tol=0.01
        ... )
        True
        >>> # wait for half a refill period for a total of 2, considering previous refill
        >>> time.sleep(0.52 * refill_rate.period_sec)
        >>> b.refill()  # should add 1 token
        >>> b.is_empty, b._current
        (False, 2)
        >>> (b._last_refill_ns - last_update) >= 2 * refill_rate.period_ns
        True
        >>>

        :return: None
        """
        refill_needed = self._current < self._volume
        if not refill_needed:
            self._last_refill_ns = time.monotonic_ns()
            return
        time_since_last_refill = time.monotonic_ns() - self._last_refill_ns
        n_tokens = int(time_since_last_refill / self._refill_rate.period_ns)
        self._current = min(self._current + n_tokens, self._volume)
        # Offset to not lose tokens. Might be significant with slow refill rates.
        self._last_refill_ns += n_tokens * self._refill_rate.period_ns

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"volume={self._volume}, "
            f"refill_rate={self._refill_rate}"
            f")"
        )

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tokens={self._current}, "
            f"volume={self._volume}, "
            f"refill_rate={self._refill_rate}"
            f")"
        )


class RateLimiter:
    """
    Basic rate limiter

    Provides ``allow`` method to check if rate limit exceeded or not.
    Client code is responsible for handling actual request.

    >>> rl = RateLimiter()  # 1 rps

    loop to simulate incoming requests. Append result for each request.
    Every second request should be rejected.
    >>> rps = Rate(2)  # 2 rps
    >>> acceptance: list[bool] = []
    >>> for _ in range(5):
    ...     acceptance.append(rl.allow())
    ...     time.sleep(rps.period_sec)
    >>> acceptance
    [True, False, True, False, True]
    """
    _bucket: Final[Bucket]

    def __init__(self, rps: float = 1.0) -> None:
        self._bucket = Bucket(1, Rate.from_frequency(rps))

    def allow(self) -> bool:
        """
        This method should be called every time client code receives a request.
        Updates internal timer on each call.

        :return: True if rate limit is not exceeded, False otherwise
        """
        self._bucket.refill()

        if self._bucket.is_empty:
            return False

        self._bucket.remove()

        return True
