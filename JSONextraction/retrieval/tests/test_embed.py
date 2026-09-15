"""The real `retrieval.embed.Embedder`, against a fake Mistral client.

Every other test replaces the Embedder wholesale, so without these its own
rules — which failures are retried, the vector-width invariant, the count
check — ran only against the live API. The client is faked at the
`mistralai` boundary, so no key and no tokens are needed.

    python -m unittest discover -s retrieval/tests
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from mistralai.client.errors import SDKError
from tenacity import wait_none

from retrieval.embed import Embedder, is_retryable


def _sdk_error(status: int) -> SDKError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.mistral.ai/v1/embeddings"))
    return SDKError(f"HTTP {status}", response)


def _response(vectors: list[list[float]], tokens: int = 7):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v) for v in vectors],
        usage=SimpleNamespace(total_tokens=tokens),
    )


class RetryPolicyTests(unittest.TestCase):
    def test_rate_limit_and_server_faults_are_retried(self) -> None:
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(is_retryable(_sdk_error(status)))

    def test_client_errors_fail_immediately(self) -> None:
        """A bad key retried six times hides the real message behind a minute of
        backoff, and still fails."""
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(is_retryable(_sdk_error(status)))

    def test_timeouts_and_connection_errors_are_retried(self) -> None:
        self.assertTrue(is_retryable(TimeoutError()))
        self.assertTrue(is_retryable(ConnectionError()))

    def test_programming_errors_are_not_retried(self) -> None:
        self.assertFalse(is_retryable(ValueError("bug")))
        self.assertFalse(is_retryable(RuntimeError("dimension changed")))


class EmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("retrieval.embed.Mistral")
        self.client = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.embedder = Embedder(api_key="fake", model="mistral-embed")
        # No real sleeping between retry attempts in tests. The retry object
        # belongs to the decorated method, shared by every instance, so restore
        # it rather than leak zero-wait retries into other tests.
        retrying = Embedder._call.retry
        original_wait = retrying.wait
        retrying.wait = wait_none()
        self.addCleanup(setattr, retrying, "wait", original_wait)

    def test_vectors_and_token_usage_are_returned_and_accumulated(self) -> None:
        self.client.embeddings.create.return_value = _response([[0.1, 0.2], [0.3, 0.4]], tokens=5)
        self.assertEqual(self.embedder.embed(["a", "b"]), [[0.1, 0.2], [0.3, 0.4]])
        self.embedder.embed(["a", "b"])
        self.assertEqual(self.embedder.total_tokens, 10)
        self.assertEqual(self.embedder.dimension, 2)
        self.client.embeddings.create.assert_called_with(model="mistral-embed", inputs=["a", "b"])

    def test_a_rate_limit_is_retried_until_it_succeeds(self) -> None:
        self.client.embeddings.create.side_effect = [_sdk_error(429), _sdk_error(503), _response([[1.0, 0.0]])]
        self.assertEqual(self.embedder.embed(["a"]), [[1.0, 0.0]])
        self.assertEqual(self.client.embeddings.create.call_count, 3)

    def test_a_bad_key_is_raised_on_the_first_attempt(self) -> None:
        self.client.embeddings.create.side_effect = _sdk_error(401)
        with self.assertRaises(SDKError):
            self.embedder.embed(["a"])
        self.assertEqual(self.client.embeddings.create.call_count, 1)

    def test_retries_are_bounded(self) -> None:
        self.client.embeddings.create.side_effect = _sdk_error(429)
        with self.assertRaises(SDKError):
            self.embedder.embed(["a"])
        self.assertEqual(self.client.embeddings.create.call_count, 6)

    def test_a_dimension_change_mid_run_is_refused(self) -> None:
        """The failure the collection naming cannot catch within one run: the
        model changing underneath a job that has already written vectors."""
        self.client.embeddings.create.side_effect = [_response([[0.1, 0.2]]), _response([[0.1, 0.2, 0.3]])]
        self.embedder.embed(["a"])
        with self.assertRaises(RuntimeError) as caught:
            self.embedder.embed(["b"])
        self.assertIn("dimension changed mid-run", str(caught.exception))

    def test_a_short_response_is_refused_rather_than_misaligned(self) -> None:
        """Pairing N texts with N-1 vectors would attach every later vector to
        the wrong row, silently."""
        self.client.embeddings.create.return_value = _response([[0.1, 0.2]])
        with self.assertRaises(RuntimeError) as caught:
            self.embedder.embed(["a", "b"])
        self.assertIn("refusing to guess", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
